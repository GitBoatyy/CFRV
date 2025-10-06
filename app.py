from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3, calendar
from datetime import datetime, date

app = Flask(__name__)
app.secret_key = "supersecretkey"
DB = "db.sqlite3"

# default list of sites to seed into a fresh database
DEFAULT_SITES = [
    *(f"Site {i}" for i in range(1, 11)),
    "Dahlia East",
    "Dahlia West",
]

# -----------------------
# DB helpers / migrations
# -----------------------
def get_conn():
    return sqlite3.connect(DB)

def query_db(q, args=(), one=False):
    conn = get_conn()
    conn.row_factory = None
    cur = conn.cursor()
    cur.execute(q, args)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    return (rows[0] if rows else None) if one else rows

def table_has_column(table, column):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    conn.close()
    return column in cols

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS sites (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS reservations (
        id INTEGER PRIMARY KEY,
        site_id INTEGER,
        guest_name TEXT,
        phone TEXT,
        email TEXT,
        arrival_date TEXT,
        departure_date TEXT,
        status TEXT,
        rv_size TEXT,
        num_campers INTEGER,
        paid TEXT,
        notes TEXT,
        num_adults INTEGER,
        num_children INTEGER,
        FOREIGN KEY(site_id) REFERENCES sites(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT
    )""")

    conn.commit()

    # seed
    existing_site_names = {row[0] for row in c.execute("SELECT name FROM sites").fetchall()}
    for name in DEFAULT_SITES:
        if name not in existing_site_names:
            c.execute("INSERT INTO sites (name) VALUES (?)", (name,))
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        c.execute("INSERT INTO users (username,password) VALUES (?,?)", ("admin","admin"))
    conn.commit()

    # migrations
    if not table_has_column("reservations", "notes"):
        c.execute("ALTER TABLE reservations ADD COLUMN notes TEXT")
    if not table_has_column("reservations", "num_adults"):
        c.execute("ALTER TABLE reservations ADD COLUMN num_adults INTEGER")
    if not table_has_column("reservations", "num_children"):
        c.execute("ALTER TABLE reservations ADD COLUMN num_children INTEGER")
    conn.commit()
    conn.close()

# -----------------------
# Availability / overlap
# -----------------------
def is_available(site_id, start, end, exclude_res_id=None):
    overlaps = query_db("""
        SELECT id FROM reservations
        WHERE site_id=?
        AND status IN ('tentative','confirmed')
        AND NOT (departure_date <= ? OR arrival_date >= ?)
    """,(site_id,start,end))
    if exclude_res_id:
        overlaps = [r for r in overlaps if r[0] != exclude_res_id]  # <-- ignore self
    return len(overlaps)==0


def parse_d(dstr):
    y, m, dd = map(int, dstr.split("-"))
    return date(y, m, dd)

def build_row_cells(days, site_res_list):
    res = []
    for r in site_res_list:
        res.append({
            **r,
            "arrival_d": parse_d(r["arrival"]),
            "departure_d": parse_d(r["departure"])
        })
    res.sort(key=lambda x: x["arrival_d"])

    cells, i = [], 0
    while i < len(days):
        d = days[i]
        booked = None
        for r in res:
            if r["arrival_d"] <= d < r["departure_d"]:
                booked = r
                break
        if booked:
            span, j = 0, i
            while j < len(days) and days[j] < booked["departure_d"]:
                span += 1
                j += 1
            cells.append({"type": "res", "span": span, "res": booked})
            i += span
        else:
            cells.append({"type": "free", "span": 1})
            i += 1
    return cells

# -----------------------
# Auth
# -----------------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = query_db(
            "SELECT * FROM users WHERE username=? AND password=?",
            (request.form["username"], request.form["password"]), one=True
        )
        if user:
            session["user_id"] = user[0]
            return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear
    return redirect(url_for("availability"))

# -----------------------
# Public availability (read-only)
# -----------------------
@app.route("/")
def home():
    return redirect(url_for("availability"))

@app.route("/availability")
def availability():
    today = date.today()
    year = request.args.get("year", type=int, default=today.year)
    month = request.args.get("month", type=int, default=today.month)

    # no browsing to past months for public
    if (year, month) < (today.year, today.month):
        year, month = today.year, today.month

    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]

    sites = query_db("SELECT id,name FROM sites ORDER BY id")
    reservations = query_db("SELECT site_id, arrival_date, departure_date FROM reservations")

    grid = {}
    for sid, _sname in sites:
        row = []
        for d in days:
            booked = False
            for r in reservations:
                if r[0] == sid and r[1] <= d.isoformat() < r[2]:
                    booked = True
                    break
            row.append("booked" if booked else "free")
        grid[sid] = row

    prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
    next_y, next_m = (year + 1, 1) if month == 12 else (year, month + 1)

    return render_template(
        "availability.html",
        year=year, month=month, days=days,
        sites=sites, grid=grid, today=today,
        prev_year=prev_y, prev_month=prev_m,
        next_year=next_y, next_month=next_m
    )

# -----------------------
# Staff dashboard
# -----------------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    # current selection
    year = int(request.args.get("year", datetime.now().year))
    month = int(request.args.get("month", datetime.now().month))

    # prev/next helpers
    def prev_ym(y, m):
        return (y-1, 12) if m == 1 else (y, m-1)
    def next_ym(y, m):
        return (y+1, 1) if m == 12 else (y, m+1)

    prev_year, prev_month = prev_ym(year, month)
    next_year, next_month = next_ym(year, month)

    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]

    sites = query_db("SELECT id,name FROM sites ORDER BY id")
    start, end = days[0].isoformat(), days[-1].isoformat()

    reservations = query_db("""
        SELECT r.id, r.site_id, r.guest_name, r.phone, r.email,
               r.arrival_date, r.departure_date, r.status,
               r.rv_size, r.num_adults, r.num_children, r.paid, r.notes
        FROM reservations r
        WHERE NOT (r.departure_date < ? OR r.arrival_date > ?)
    """, (start, end))

    res_map = {}
    for r in reservations:
        res_map.setdefault(r[1], []).append({
            "id": r[0],
            "site_id": r[1],
            "guest": r[2],
            "phone": r[3],
            "email": r[4],
            "arrival": r[5],
            "departure": r[6],
            "status": r[7],
            "rv_size": r[8],
            "num_adults": r[9] or 0,
            "num_children": r[10] or 0,
            "paid": r[11],
            "notes": r[12] or ""
        })

    grid = {sid: build_row_cells(days, res_map.get(sid, [])) for sid, _ in sites}

    return render_template(
        "dashboard.html",
        sites=sites,
        days=days,
        grid=grid,
        year=year,
        month=month,
        prev_year=prev_year,
        prev_month=prev_month,
        next_year=next_year,
        next_month=next_month
    )


# -----------------------
# CRUD API
# -----------------------
@app.route("/api/reservation/add", methods=["POST"])
def add_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()

    site_id = int(data["site_id"])
    start = data["arrival"]
    end = data["departure"]

    if not is_available(site_id, start, end):
        return jsonify({"ok": False, "error": "Site not available"})

    query_db("""
        INSERT INTO reservations
        (site_id, guest_name, phone, email, arrival_date, departure_date,
         status, rv_size, num_campers, paid, notes, num_adults, num_children)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (site_id, data["guest"], data.get("phone"), data.get("email"),
          start, end, data.get("status","tentative"), data.get("rv_size"),
          data.get("num_campers"), data.get("paid","no"),
          data.get("notes",""), data.get("num_adults"), data.get("num_children")))
    return jsonify({"ok": True})

@app.route("/api/reservation/update", methods=["POST"])
def update_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()
    res_id = int(data["id"])
    site_id = int(data["site_id"])
    start = data["arrival"]
    end = data["departure"]

    if not is_available(site_id, start, end, exclude_res_id=res_id):
        return jsonify({"ok": False, "error": "Site not available"})

    query_db("""
        UPDATE reservations
           SET guest_name=?,
               phone=?,
               email=?,
               arrival_date=?,
               departure_date=?,
               site_id=?,
               status=?,
               rv_size=?,
               num_campers=?,
               paid=?,
               notes=?,
               num_adults=?,
               num_children=?
         WHERE id=?
    """, (data["guest"], data.get("phone"), data.get("email"),
          start, end, site_id, data.get("status","tentative"),
          data.get("rv_size"), data.get("num_campers"),
          data.get("paid","no"), data.get("notes",""),
          data.get("num_adults"), data.get("num_children"),
          res_id))
    return jsonify({"ok": True})

@app.route("/api/reservation/move", methods=["POST"])
def move_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()
    res_id = int(data["id"])
    site_id = int(data["site_id"])
    new_start = data["arrival"]
    new_end = data["departure"]

    if not is_available(site_id, new_start, new_end, exclude_res_id=res_id):
        return jsonify({"ok": False, "error": "Destination not available"})

    query_db("UPDATE reservations SET site_id=?, arrival_date=?, departure_date=? WHERE id=?",
             (site_id, new_start, new_end, res_id))
    return jsonify({"ok": True})

@app.route("/api/reservation/delete", methods=["POST"])
def delete_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()
    res_id = data.get("id")
    if not res_id:
        return jsonify({"ok": False, "error": "Missing id"})
    query_db("DELETE FROM reservations WHERE id=?", (res_id,))
    return jsonify({"ok": True})

@app.route("/api/reservation/optimize", methods=["POST"])
def optimize_reservations():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})

    site_rows = query_db("SELECT id FROM sites ORDER BY id ASC")
    site_ids = [row[0] for row in site_rows]
    if not site_ids:
        return jsonify({"ok": False, "error": "No sites configured"})

    res_rows = query_db(
        """
        SELECT id, site_id, arrival_date, departure_date
        FROM reservations
        WHERE arrival_date IS NOT NULL AND departure_date IS NOT NULL
        ORDER BY arrival_date ASC, departure_date ASC, id ASC
        """
    )

    reservations = []
    for rid, site_id, arr, dep in res_rows:
        try:
            arrival_d = parse_d(arr)
            departure_d = parse_d(dep)
        except Exception:
            continue
        if departure_d <= arrival_d:
            continue
        reservations.append({
            "id": rid,
            "current_site": site_id,
            "arrival": arrival_d,
            "departure": departure_d,
        })

    if not reservations:
        return jsonify({"ok": True, "moved": 0})

    site_available = {sid: date.min for sid in site_ids}
    moves = []

    for res in reservations:
        ordered_sites = sorted(site_ids, key=lambda sid: (site_available[sid], sid))
        target_site = None
        for sid in ordered_sites:
            if site_available[sid] <= res["arrival"]:
                target_site = sid
                break
        if target_site is None:
            return jsonify({
                "ok": False,
                "error": "Optimization failed: overlapping reservations exceed available sites."
            })

        site_available[target_site] = res["departure"]
        if target_site != res["current_site"]:
            moves.append((target_site, res["id"]))

    if moves:
        conn = get_conn()
        cur = conn.cursor()
        for site_id, res_id in moves:
            cur.execute("UPDATE reservations SET site_id=? WHERE id=?", (site_id, res_id))
        conn.commit()
        conn.close()

    return jsonify({"ok": True, "moved": len(moves)})

# -----------------------
# Export (month or range)
# -----------------------
def month_iter(y1, m1, y2, m2):
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1

@app.route("/export")
def export_reservations():
    if "user_id" not in session:
        return redirect(url_for("login"))

    mode = request.args.get("scope", "month")  # "month" or "range"
    today = date.today()
    year = int(request.args.get("year", today.year))
    month = int(request.args.get("month", today.month))

    # default month
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    start_m, end_m = days[0].isoformat(), days[-1].isoformat()

    from_month = request.args.get("from", default=None)  # 'YYYY-MM'
    to_month = request.args.get("to", default=None)      # 'YYYY-MM'

    sites = query_db("SELECT id,name FROM sites ORDER BY id ASC")
    sections = []

    if mode == "range" and from_month and to_month:
        fy, fm = map(int, from_month.split("-"))
        ty, tm = map(int, to_month.split("-"))
        for (yy, mm) in month_iter(fy, fm, ty, tm):
            nd = calendar.monthrange(yy, mm)[1]
            dlist = [date(yy, mm, d) for d in range(1, nd + 1)]
            s_iso, e_iso = dlist[0].isoformat(), dlist[-1].isoformat()

            rows = query_db("""
                SELECT r.id, r.site_id, r.guest_name, r.phone, r.email,
                       r.arrival_date, r.departure_date, r.status,
                       r.rv_size, r.num_campers, r.paid, r.notes,
                       r.num_adults, r.num_children
                FROM reservations r
                WHERE NOT (r.departure_date < ? OR r.arrival_date > ?)
                ORDER BY r.site_id ASC, r.arrival_date
            """, (s_iso, e_iso))

            res_map, all_res_list = {}, []
            for r in rows:
                rd = {
                    "id": r[0], "site_id": r[1], "guest": r[2],
                    "phone": r[3], "email": r[4],
                    "arrival": r[5], "departure": r[6], "status": r[7],
                    "rv_size": r[8], "num_campers": r[9], "paid": r[10],
                    "notes": r[11] or "",
                    "num_adults": r[12] if r[12] is not None else 0,
                    "num_children": r[13] if r[13] is not None else 0
                }
                res_map.setdefault(r[1], []).append(rd)
                all_res_list.append(rd)

            grid = {}
            for sid, _sname in sites:
                grid[sid] = build_row_cells(dlist, res_map.get(sid, []))
            sections.append({
                "year": yy, "month": mm, "days": dlist,
                "grid": grid, "all_res_list": all_res_list
            })
    else:
        rows = query_db("""
            SELECT r.id, r.site_id, r.guest_name, r.phone, r.email,
                   r.arrival_date, r.departure_date, r.status,
                   r.rv_size, r.num_campers, r.paid, r.notes,
                   r.num_adults, r.num_children
            FROM reservations r
            WHERE NOT (r.departure_date < ? OR r.arrival_date > ?)
            ORDER BY r.site_id ASC, r.arrival_date
        """, (start_m, end_m))

        res_map, all_res_list = {}, []
        for r in rows:
            rd = {
                "id": r[0], "site_id": r[1], "guest": r[2],
                "phone": r[3], "email": r[4],
                "arrival": r[5], "departure": r[6], "status": r[7],
                "rv_size": r[8], "num_campers": r[9], "paid": r[10],
                "notes": r[11] or "",
                "num_adults": r[12] if r[12] is not None else 0,
                "num_children": r[13] if r[13] is not None else 0
            }
            res_map.setdefault(r[1], []).append(rd)
            all_res_list.append(rd)

        grid = {}
        for sid, _sname in sites:
            grid[sid] = build_row_cells(days, res_map.get(sid, []))
        sections.append({
            "year": year, "month": month, "days": days,
            "grid": grid, "all_res_list": all_res_list
        })

    return render_template(
        "export.html",
        scope=mode,
        year=year, month=month,
        sites=sites,
        sections=sections
    )

# -----------------------
# boot
# -----------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
