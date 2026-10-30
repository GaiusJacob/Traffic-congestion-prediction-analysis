"""
engine.py - the traffic "world".

Everything that moves lives here (the browser only draws what this file computes).

Key ideas
---------
* Time is measured in HOURS.  One simulation step = DT hours (0.02 h = 1.2 min).
* The map is a road network.  Every road has two directions ("directed edges").
  Each directed edge holds a queue of cars, so jams can build up and spill back.
* Cars ENTER from outside the map through gateway nodes (W1, N1, E1 ...), drive to
  a different gateway, and LEAVE the map there.  New cars keep arriving forever
  (rate follows a morning / evening rush-hour profile).
* You can schedule any number of accidents (on a junction or on a road).
  - 4 hours before an accident starts, a WARNING is issued.
  - When the accident starts, every car whose route crosses it is classified:
        ETA to the accident <  4 h  -> STUCK on the roads leading to it (queue)
        ETA to the accident >= 4 h  -> REROUTED onto different roads
  - Cars that enter the map after the warning plan their trip around the accident.
* An optional forecaster (the Random Forest in forecast_model.py) predicts the load
  of every road for the next 4 hours; routing uses it to avoid roads that will jam.
"""
import math
import random
import heapq
from collections import deque

# ----------------------------------------------------------------- constants
PX_PER_HOUR = 100.0        # map pixels a car covers in one hour of free-flow driving
DT = 0.02                  # simulation step (hours)
WARNING_LEAD_H = 4.0       # warning is issued this many hours before an accident
SAT_FLOW = 26.0            # max vehicles per hour that can leave one directed road
STORAGE_PER_HOUR = 30      # vehicles that fit on a directed road per hour of length
BARRIER_POS = 0.55         # where a road accident sits along the road (0..1)
START_HOUR = 0.0           # clock time (hours) when the simulation starts (00:00)
WARMUP_H = 8.0             # hidden warm-up so the map is already full of cars at 08:00

GRID_X = [200, 400, 600, 800]
GRID_Y = [130, 265, 400, 535]
JUNCTION_NAMES = {
    "A1": "North Gate", "A2": "Market Crossing", "A3": "Old Town", "A4": "Airport Road",
    "B1": "West End", "B2": "Central Junction", "B3": "City Square", "B4": "East Terminal",
    "C1": "River Junction", "C2": "University", "C3": "Hospital", "C4": "Harbour",
    "D1": "South Gate", "D2": "Rail Station", "D3": "Industrial Park", "D4": "Stadium",
}
# gateway id -> (x, y, junction it connects to, vehicles/hour entering at profile=1)
GATES = {
    "W1": (50, 130, "A1", 8), "W2": (50, 400, "C1", 7),
    "E1": (950, 265, "B4", 8), "E2": (950, 535, "D4", 6),
    "N1": (400, 35, "A2", 7), "N2": (800, 35, "A4", 6),
    "S1": (200, 610, "D1", 6), "S2": (600, 610, "D3", 7),
}
REMOVED_ROADS = {("C1", "C2")}                      # a "river" with no bridge here
DIAGONALS = [("A2", "B3"), ("C2", "D3")]            # shortcut roads


def fmt_clock(t):
    t = t + 1e-6
    day = int(t // 24) + 1
    h = int(t % 24)
    m = int((t % 1) * 60 + 1e-6)
    return f"Day {day} {h:02d}:{m:02d}"


def demand_profile(hod):
    """Relative traffic demand by hour of day: morning + evening rush hours."""
    return (0.35
            + 0.95 * math.exp(-((hod - 8.0) ** 2) / (2 * 1.4 ** 2))
            + 1.05 * math.exp(-((hod - 17.5) ** 2) / (2 * 1.7 ** 2)))


# ------------------------------------------------------------------- network
class Edge:
    __slots__ = ("idx", "frm", "to", "road", "L", "cap", "gap", "rev")


class Network:
    def __init__(self):
        self.nodes = {}
        for r, row in enumerate("ABCD"):
            for c in range(4):
                nid = f"{row}{c + 1}"
                self.nodes[nid] = dict(id=nid, name=JUNCTION_NAMES[nid], x=GRID_X[c], y=GRID_Y[r], kind="junction")
        for gid, (x, y, _, _) in GATES.items():
            self.nodes[gid] = dict(id=gid, name=f"Gateway {gid}", x=x, y=y, kind="gate")

        roads = []
        for row in "ABCD":
            for c in range(3):
                a, b = f"{row}{c + 1}", f"{row}{c + 2}"
                if (a, b) not in REMOVED_ROADS:
                    roads.append((a, b))
        for c in range(1, 5):
            for r in range(3):
                roads.append((f"{'ABCD'[r]}{c}", f"{'ABCD'[r + 1]}{c}"))
        roads += DIAGONALS
        roads += [(g, v[2]) for g, v in GATES.items()]
        self.roads = roads

        self.edges, self.out, self.inc = [], {n: [] for n in self.nodes}, {n: [] for n in self.nodes}
        for ri, (a, b) in enumerate(roads):
            na, nb = self.nodes[a], self.nodes[b]
            t0 = max(0.25, math.hypot(na["x"] - nb["x"], na["y"] - nb["y"]) / PX_PER_HOUR)
            for frm, to in ((a, b), (b, a)):
                e = Edge()
                e.idx, e.frm, e.to, e.road, e.L = len(self.edges), frm, to, ri, t0
                e.cap = max(5, round(t0 * STORAGE_PER_HOUR))
                e.gap = t0 / e.cap
                e.rev = -1
                self.edges.append(e)
                self.out[frm].append(e.idx)
                self.inc[to].append(e.idx)
            self.edges[-2].rev, self.edges[-1].rev = len(self.edges) - 1, len(self.edges) - 2
        self.road_edges = {}
        for e in self.edges:
            self.road_edges.setdefault(e.road, []).append(e.idx)

        # hop distances between nodes (for model features)
        self.hops = {}
        for s in self.nodes:
            dist, q = {s: 0}, deque([s])
            while q:
                u = q.popleft()
                for ei in self.out[u]:
                    v = self.edges[ei].to
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.hops[s] = dist
        self.up_edges = [[i for i in self.inc[e.frm] if i != e.rev] for e in self.edges]
        self.down_edges = [[i for i in self.out[e.to] if i != e.rev] for e in self.edges]

    def edge_name(self, ei):
        e = self.edges[ei]
        return f"{e.frm}->{e.to}"

    def to_json(self):
        return dict(
            nodes=list(self.nodes.values()),
            roads=[dict(idx=i, a=a, b=b) for i, (a, b) in enumerate(self.roads)],
            edges=[dict(idx=e.idx, frm=e.frm, to=e.to, road=e.road, L=round(e.L, 3), cap=e.cap) for e in self.edges],
            warning_lead=WARNING_LEAD_H,
        )


# ---------------------------------------------------------------------- cars
class Car:
    __slots__ = ("id", "route", "k", "pos", "v", "dest", "rerouted", "stuck", "t_in", "t_free", "wait", "vel")


# -------------------------------------------------------------------- engine
class Engine:
    def __init__(self, net, seed=1, demand=1.0, forecaster=None, mode="reactive",
                 use_rf=True, warmup=True, start_hour=START_HOUR):
        self.net = net
        self.rng = random.Random(seed)
        self.demand = demand
        self.forecaster = forecaster
        self.mode = mode                 # 'reactive' or 'early'
        self.use_rf = use_rf
        self.t = start_hour
        E = len(net.edges)
        self.veh = [[] for _ in range(E)]
        self.tok = [0.0] * E             # discharge tokens (how many cars may leave this step)
        self.btok = [0.0] * E            # tokens for crossing a road accident
        self.sf = [1.0] * E              # current speed factor of every road (1 = free flow)
        self.blocks = {}                 # edge idx -> (kind, severity) while an accident is active
        self.pending = {g: 0 for g in GATES}
        self.accidents = []
        self.next_acc_id = 1
        self.next_car_id = 1
        self.events = []
        self.entered = self.exited = self.rerouted_total = 0
        self.delays = deque(maxlen=200)
        self.hist = deque(maxlen=3)      # hourly load snapshots (t-3, t-2, t-1)
        self.pred = None                 # forecast: list[edge][4]  (load ratio at +1..+4 h)
        self.pred_clock = None
        self.hour_hooks = []
        self.dest_w = {}
        for g in GATES:
            others = [d for d in GATES if d != g]
            self.dest_w[g] = (others, [(1 + net.hops[g][d]) * GATES[d][3] for d in others])
        if warmup:
            self.run(WARMUP_H)
            self.events.clear()
            self.entered = self.exited = self.rerouted_total = 0
            self.delays.clear()

    # ------------------------------------------------------------- logging
    def log(self, text, kind=""):
        self.events.append(dict(clock=fmt_clock(self.t), text=text, kind=kind))

    # ------------------------------------------------------------ accidents
    def label_of(self, kind, target):
        if kind == "node":
            return f"{target} {self.net.nodes[target]['name']}"
        a, b = self.net.roads[target]
        return f"road {a}-{b}"

    def add_accident(self, kind, target, start, duration, severity):
        net = self.net
        if kind == "node":
            if target not in net.nodes or net.nodes[target]["kind"] != "junction":
                raise ValueError("Pick a junction (not a gateway).")
            aff = set(net.inc[target])
            nodes = [target]
        elif kind == "road":
            target = int(target)
            if not 0 <= target < len(net.roads):
                raise ValueError("Unknown road.")
            aff = set(net.road_edges[target])
            nodes = list(net.roads[target])
        else:
            raise ValueError("kind must be 'node' or 'road'.")
        start = max(float(start), self.t + 0.05)
        duration = min(max(float(duration), 0.5), 12.0)
        severity = min(max(float(severity), 0.1), 1.0)
        acc = dict(id=self.next_acc_id, kind=kind, target=target, label=self.label_of(kind, target),
                   start=start, end=start + duration, dur=duration, sev=severity,
                   status="scheduled", aff=aff, nodes=nodes, n_stuck=0, n_rerouted=0)
        self.next_acc_id += 1
        self.accidents.append(acc)
        self.log(f"Accident #{acc['id']} scheduled at {acc['label']} for {fmt_clock(start)} "
                 f"({duration:g} h, {'full closure' if severity >= 0.999 else f'{int(severity * 100)}% capacity loss'}).", "ok")
        return acc

    def remove_accident(self, acc_id):
        for a in self.accidents:
            if a["id"] == acc_id and a["status"] in ("scheduled", "warning", "active"):
                a["status"] = "cancelled"
                self._release(a)
                self._recompute_blocks()
                self.log(f"Accident #{acc_id} cancelled.", "ok")
                return True
        return False

    def _release(self, acc):
        for vs in self.veh:
            for c in vs:
                if c.stuck == acc["id"]:
                    c.stuck = 0

    def _recompute_blocks(self):
        self.blocks = {}
        for a in self.accidents:
            if a["status"] == "active":
                for ei in a["aff"]:
                    cur = self.blocks.get(ei)
                    if cur is None or a["sev"] > cur[1]:
                        self.blocks[ei] = (a["kind"], a["sev"])

    def _update_accidents(self):
        changed = False
        for a in self.accidents:
            if a["status"] == "scheduled" and self.t >= a["start"] - WARNING_LEAD_H:
                a["status"] = "warning"
                self.log(f"EARLY WARNING: accident #{a['id']} expected at {a['label']} at "
                         f"{fmt_clock(a['start'])}. Traffic forecast now includes it.", "warn")
                if self.mode == "early":
                    n = self._respond(a, "warning")
                    self.log(f"Early rerouting: {n} vehicles heading for #{a['id']} were sent on different roads.", "ok")
            if a["status"] == "warning" and self.t >= a["start"]:
                a["status"] = "active"
                changed = True
                self._recompute_blocks()
                self._respond(a, "onset")
                self.log(f"ACCIDENT #{a['id']} ACTIVE at {a['label']}: {a['n_stuck']} vehicles stuck on the approach "
                         f"roads (< {WARNING_LEAD_H:g} h away), {a['n_rerouted']} rerouted (>= {WARNING_LEAD_H:g} h away).", "alert")
            if a["status"] == "active" and self.t >= a["end"]:
                a["status"] = "cleared"
                changed = True
                self._release(a)
                self.log(f"Accident #{a['id']} at {a['label']} cleared. The queue is discharging.", "ok")
        if changed:
            self._recompute_blocks()

    # -------------------------------------------------------------- routing
    def _tt(self, ei, offset):
        """Expected travel time of a directed road, `offset` hours from now."""
        e = self.net.edges[ei]
        if self.use_rf and self.pred is not None and offset >= 0.5:
            rho = self.pred[ei][min(4, max(1, round(offset))) - 1]
        else:
            rho = len(self.veh[ei]) / e.cap
        rho = min(max(rho, 0.0), 1.3)
        return e.L * (1 + 1.5 * rho ** 3)

    def _known(self):
        """Accidents drivers know about. Reactive drivers only know accidents that already
        happened; in 'early' mode drivers also act on the 4-hour warning."""
        seen = ("warning", "active") if self.mode == "early" else ("active",)
        return [a for a in self.accidents if a["status"] in seen]

    def _dijkstra(self, src, dest, t_abs, use_acc):
        net = self.net
        known = self._known() if use_acc else []
        shift = t_abs - self.t
        best, prev, pq = {src: 0.0}, {}, [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > best.get(u, 1e9):
                continue
            if u == dest:
                break
            if u != src and net.nodes[u]["kind"] == "gate":
                continue                                   # gateways are never shortcuts
            for ei in net.out[u]:
                tt = self._tt(ei, d + shift)
                mult, skip, ta = 1.0, False, t_abs + d
                for a in known:
                    if ei in a["aff"] and ta < a["end"] and ta + tt > a["start"]:
                        if a["sev"] >= 0.999:
                            skip = True
                            break
                        mult *= 1 + 10 * a["sev"] ** 2
                if skip:
                    continue
                v = net.edges[ei].to
                nd = d + tt * mult
                if nd < best.get(v, 1e9):
                    best[v] = nd
                    prev[v] = ei
                    heapq.heappush(pq, (nd, v))
        if dest not in prev:
            return None
        path, n = [], dest
        while n != src:
            ei = prev[n]
            path.append(ei)
            n = net.edges[ei].frm
        path.reverse()
        return path

    def plan(self, src, dest, t_abs):
        """Route avoiding known accidents if possible. Returns (path, avoided_all)."""
        p = self._dijkstra(src, dest, t_abs, True)
        if p is not None:
            return p, True
        return self._dijkstra(src, dest, t_abs, False), False

    def _eta_to(self, c, acc):
        """Hours until car `c` reaches the accident on its current route (None if it never does)."""
        edges = self.net.edges
        eta = 0.0
        for j in range(c.k, len(c.route)):
            ei = c.route[j]
            e = edges[ei]
            tt = e.L * (1 + 1.5 * min(len(self.veh[ei]) / e.cap, 1.3) ** 3)
            f0 = c.pos / e.L if j == c.k else 0.0
            if ei in acc["aff"]:
                point = 1.0 if acc["kind"] == "node" else BARRIER_POS
                if point <= f0:
                    return None                            # already passed the accident
                return eta + (point - f0) * tt
            eta += tt * (1 - f0)
        return None

    def _reroute(self, c):
        e = self.net.edges[c.route[c.k]]
        if e.to == c.dest:
            return False
        wait = (e.L - c.pos) / max(0.05, c.v * self.sf[e.idx])
        path, ok = self.plan(e.to, c.dest, self.t + wait)
        if not ok or path is None:
            return False
        c.route = c.route[:c.k + 1] + path
        if not c.rerouted:
            c.rerouted = True
            self.rerouted_total += 1
        return True

    def _respond(self, acc, phase):
        now, n_re, n_st = self.t, 0, 0
        for vs in self.veh:
            for c in vs:
                eta = self._eta_to(c, acc)
                if eta is None or now + eta >= acc["end"]:
                    continue
                committed = c.route[c.k] in acc["aff"]
                if phase == "warning":
                    if now + eta >= acc["start"] and not committed and self._reroute(c):
                        n_re += 1
                    continue
                # phase == "onset": the 4-hour rule
                if committed or eta < WARNING_LEAD_H:
                    c.stuck = acc["id"]
                    n_st += 1
                elif self._reroute(c):
                    n_re += 1
                else:
                    c.stuck = acc["id"]
                    n_st += 1
        if phase == "onset":
            acc["n_stuck"] = n_st
        acc["n_rerouted"] += n_re
        return n_re

    # ----------------------------------------------------------- simulation
    def loads(self):
        edges = self.net.edges
        return [len(self.veh[i]) / edges[i].cap for i in range(len(edges))]

    def _spawn(self, dt):
        net = self.net
        hod = self.t % 24
        scale = demand_profile(hod) * self.demand
        for g, (_, _, _, rate) in GATES.items():
            if self.rng.random() < rate * scale * dt and self.pending[g] < 40:
                self.pending[g] += 1
            if self.pending[g] <= 0:
                continue
            ei = net.out[g][0]
            vs = self.veh[ei]
            if vs and vs[-1].pos < net.edges[ei].gap:
                continue                                    # entrance is full, wait outside
            others, w = self.dest_w[g]
            dest = self.rng.choices(others, w)[0]
            path, _ = self.plan(g, dest, self.t)
            self.pending[g] -= 1
            if not path:
                continue
            detour = False
            known = self._known()
            if known:                                        # would the normal route hit an accident?
                base = self._dijkstra(g, dest, self.t, False)
                if base and base != path:
                    detour = True
                    self.rerouted_total += 1
                    for a in known:
                        if any(ei in a["aff"] for ei in base):
                            a["n_rerouted"] += 1
            c = Car()
            c.id, c.route, c.k, c.pos = self.next_car_id, path, 0, 0.0
            c.v = self.rng.uniform(0.9, 1.1)
            c.dest, c.rerouted, c.stuck, c.wait, c.vel = dest, detour, 0, 0.0, 0.0
            c.t_in = self.t
            c.t_free = sum(net.edges[i].L for i in path) / c.v
            self.next_car_id += 1
            self.entered += 1
            vs.append(c)

    def _advance_edge(self, ei, dt):
        e = self.net.edges[ei]
        vs = self.veh[ei]
        blk = self.blocks.get(ei)
        sev = blk[1] if blk else 0.0
        node_blk = blk is not None and blk[0] == "node"
        road_blk = blk is not None and blk[0] == "road"
        self.tok[ei] = tok = min(self.tok[ei] + SAT_FLOW * dt * ((1 - sev) if node_blk else 1.0), 2.0)
        if road_blk:
            self.btok[ei] = bt = min(self.btok[ei] + SAT_FLOW * dt * (1 - sev), 2.0)
        else:
            bt = 0.0
        n = len(vs)
        if n == 0:
            self.sf[ei] = 1.0
            return
        rho = n / e.cap
        sf = self.sf[ei] = 1.0 / (1 + 1.5 * rho ** 3)
        L, gap = e.L, e.gap
        bp = BARRIER_POS * L if road_blk else None
        for i, c in enumerate(vs):
            new = c.pos + c.v * sf * dt
            limit = L if i == 0 else vs[i - 1].pos - gap
            if bp is not None and c.pos < bp <= new:
                if bt >= 1:
                    bt -= 1
                else:
                    new = bp - 1e-4
            if new > limit:
                new = limit
            if new < c.pos:
                new = c.pos
            c.vel = (new - c.pos) / dt / L
            c.pos = new
        if road_blk:
            self.btok[ei] = bt
        # the front car may leave the road
        c = vs[0]
        if c.pos >= L - 1e-6:
            if c.k == len(c.route) - 1:
                vs.pop(0)                                   # leaves the map at its gateway
                self.exited += 1
                self.delays.append(self.t - c.t_in - c.t_free)
            elif tok >= 1:
                nxt = c.route[c.k + 1]
                nv = self.veh[nxt]
                free = (not nv) or nv[-1].pos >= self.net.edges[nxt].gap * (1.0 if c.wait < 1.5 else 0.2)
                if free:
                    vs.pop(0)
                    c.k += 1
                    c.pos, c.wait = 0.0, 0.0
                    nv.append(c)
                    self.tok[ei] = tok - 1
                else:
                    c.wait += dt                            # gridlock safety valve

    def step(self, dt=DT):
        t_old = self.t
        self._update_accidents()
        self._spawn(dt)
        order = list(range(len(self.veh)))
        self.rng.shuffle(order)
        for ei in order:
            self._advance_edge(ei, dt)
        self.t += dt
        if math.floor(self.t) != math.floor(t_old):
            self._hourly()

    def _hourly(self):
        if self.forecaster is not None:
            try:
                self.pred = self.forecaster.predict(self)
                self.pred_clock = fmt_clock(self.t)
            except Exception as ex:                          # never break the simulation
                self.pred = None
                self.log(f"Forecast failed: {ex}", "warn")
        for hook in self.hour_hooks:
            hook(self)
        self.hist.append(self.loads())

    def advance(self, hours):
        steps = max(1, int(round(hours / DT)))
        for _ in range(steps):
            self.step()

    run = advance

    # ---------------------------------------------------------------- state
    def state(self, since=0):
        edges = self.net.edges
        cars = []
        stuck = 0
        for ei, vs in enumerate(self.veh):
            e = edges[ei]
            slow = self.sf[ei] < 0.5
            for c in vs:
                if c.stuck:
                    flag = 2
                    stuck += 1
                elif c.rerouted:
                    flag = 1
                elif slow:
                    flag = 3
                else:
                    flag = 0
                cars.append([c.id, ei, round(c.pos / e.L, 3), round(c.vel, 2), flag])
        delay = (sum(self.delays) / len(self.delays) * 60) if self.delays else 0.0
        accs = []
        for a in self.accidents:
            live = sum(1 for vs in self.veh for c in vs if c.stuck == a["id"]) if a["status"] == "active" else 0
            accs.append(dict(id=a["id"], kind=a["kind"], target=a["target"], label=a["label"], start=round(a["start"], 2),
                             end=round(a["end"], 2), sev=a["sev"], status=a["status"], n_stuck=a["n_stuck"],
                             n_rerouted=a["n_rerouted"], stuck_now=live,
                             start_clock=fmt_clock(a["start"]), end_clock=fmt_clock(a["end"])))
        return dict(
            t=round(self.t, 3), clock=fmt_clock(self.t), demand_now=round(demand_profile(self.t % 24) * self.demand, 2),
            cars=cars, loads=[round(x, 2) for x in self.loads()],
            pred=None if self.pred is None else [[round(float(x), 2) for x in row] for row in self.pred],
            pred_clock=self.pred_clock, accidents=accs,
            stats=dict(in_network=len(cars), entered=self.entered, exited=self.exited,
                       waiting=sum(self.pending.values()), stuck=stuck, rerouted=self.rerouted_total,
                       avg_delay_min=round(delay, 1)),
            events=self.events[since:], n_events=len(self.events),
            settings=dict(demand=self.demand, mode=self.mode, use_rf=self.use_rf, model=self.forecaster is not None),
        )
