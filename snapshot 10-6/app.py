from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3, calendar
from datetime import datetime, date, timedelta
from bisect import bisect_left

app = Flask(__name__)
app.secret_key = "supersecretkey"
DB = "db.sqlite3"

MAX_AVAIL_DISPLAY_DAYS = 14
MAX_AVAIL_LOOKAHEAD_DAYS = 60

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
    if not table_has_column("reservations", "site_locked"):
        c.execute("ALTER TABLE reservations ADD COLUMN site_locked INTEGER DEFAULT 0")
    # enforce rule: tentative reservations cannot be marked paid
    c.execute("UPDATE reservations SET paid='no' WHERE status IS NOT NULL AND lower(status) != 'confirmed' AND lower(COALESCE(paid,'')) = 'yes'")
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


def to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def short_site_label(name: str) -> str:
    if name.startswith("Site "):
        return name.split(" ", 1)[1]
    if name.startswith("Dahlia "):
        parts = name.split()
        if len(parts) > 1:
            return f"Dahlia {parts[1][0]}"
        return "Dahlia"
    return name

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


def get_site_rows():
    return query_db(
        "SELECT id,name FROM sites ORDER BY CASE WHEN name LIKE 'Site %' THEN CAST(substr(name,6) AS INTEGER) ELSE 9999 END, name"
    )


def find_available_site(start_iso, end_iso):
    for sid, _name in get_site_rows():
        if is_available(sid, start_iso, end_iso):
            return sid
    return None


def range_requires_split(start_iso, end_iso):
    site_rows = get_site_rows()
    if not site_rows:
        return False
    site_ids = [sid for sid, _ in site_rows]
    if any(is_available(sid, start_iso, end_iso) for sid in site_ids):
        return False

    start_date = parse_d(start_iso)
    end_date = parse_d(end_iso)
    if start_date >= end_date:
        return False
    day = start_date
    while day < end_date:
        next_day_iso = (day + timedelta(days=1)).isoformat()
        day_iso = day.isoformat()
        if not any(is_available(sid, day_iso, next_day_iso) for sid in site_ids):
            return False
        day += timedelta(days=1)
    return True


def auto_assign_site(start_iso, end_iso):
    site_rows = get_site_rows()
    site_lookup = {sid: short_site_label(name) for sid, name in site_rows}
    warnings = []

    site_id = find_available_site(start_iso, end_iso)
    if site_id is not None:
        warnings.append(f"Auto-assigned to {site_lookup.get(site_id, f'Site {site_id}')}")
        return site_id, warnings, False, None, 0

    ok, moved, error = run_optimizer_internal()
    if not ok:
        return None, warnings, False, error, 0

    site_id = find_available_site(start_iso, end_iso)
    if site_id is not None:
        if moved:
            warnings.append(f"Optimizer moved {moved} reservation(s) to create availability.")
        warnings.append(f"Auto-assigned to {site_lookup.get(site_id, f'Site {site_id}')}")
        return site_id, warnings, False, None, moved

    if range_requires_split(start_iso, end_iso):
        return None, warnings, True, None, moved

    return None, warnings, False, "No availability for requested dates.", moved


def run_optimizer_internal():
    site_rows = query_db("SELECT id,name FROM sites ORDER BY CASE WHEN name LIKE 'Site %' THEN CAST(substr(name,6) AS INTEGER) ELSE 9999 END, name")
    site_ids = [row[0] for row in site_rows]
    if not site_ids:
        return False, 0, "No sites configured"

    res_rows = query_db(
        """
        SELECT id, site_id, arrival_date, departure_date, site_locked
        FROM reservations
        WHERE arrival_date IS NOT NULL AND departure_date IS NOT NULL
        ORDER BY arrival_date ASC, departure_date ASC, id ASC
        """
    )

    locked_by_site = {sid: [] for sid in site_ids}
    unlocked = []

    for rid, site_id, arr, dep, locked_flag in res_rows:
        try:
            arrival_d = parse_d(arr)
            departure_d = parse_d(dep)
        except Exception:
            continue
        if departure_d <= arrival_d:
            continue
        entry = {
            "id": rid,
            "current_site": site_id,
            "arrival": arrival_d,
            "departure": departure_d,
            "locked": bool(locked_flag),
        }
        if entry["locked"]:
            locked_by_site.setdefault(site_id, []).append(entry)
        else:
            unlocked.append(entry)

    if not unlocked and all(not locked_by_site[sid] for sid in site_ids):
        return True, 0, None

    schedule = {sid: [] for sid in site_ids}

    for sid in site_ids:
        locked_list = sorted(locked_by_site.get(sid, []), key=lambda r: (r["arrival"], r["departure"], r["id"]))
        last_dep = date.min
        for item in locked_list:
            if item["arrival"] < last_dep:
                return False, 0, "Optimization failed: locked reservations overlap on site."
            schedule[sid].append({
                "arrival": item["arrival"],
                "departure": item["departure"],
                "id": item["id"],
                "locked": True,
            })
            last_dep = item["departure"]

    def find_position(intervals, arrival):
        lo, hi = 0, len(intervals)
        while lo < hi:
            mid = (lo + hi) // 2
            if intervals[mid]["arrival"] < arrival:
                lo = mid + 1
            else:
                hi = mid
        return lo

    site_priority = {sid: idx for idx, (sid, _name) in enumerate(site_rows)}

    unlocked.sort(key=lambda r: (
        r["arrival"],
        -((r["departure"] - r["arrival"]).days),
        site_priority.get(r["current_site"], 999),
        r["departure"],
        r["id"],
    ))

    moves = []

    for res in unlocked:
        arrival = res["arrival"]
        departure = res["departure"]
        stay_length = max((departure - arrival).days, 1)

        best_choice = None
        best_score = None

        for sid in site_ids:
            intervals = schedule[sid]
            pos = find_position(intervals, arrival)
            prev_interval = intervals[pos - 1] if pos > 0 else None
            next_interval = intervals[pos] if pos < len(intervals) else None

            if prev_interval and arrival < prev_interval["departure"]:
                continue
            if next_interval and departure > next_interval["arrival"]:
                continue

            prev_gap = (arrival - prev_interval["departure"]).days if prev_interval else 10 ** 6
            next_gap = (next_interval["arrival"] - departure).days if next_interval else 10 ** 6
            same_site = 0 if sid == res["current_site"] else 1
            category = site_priority.get(sid, 999)
            category_penalty = category * stay_length
            score = (prev_gap, same_site, category_penalty, next_gap, category, sid)

            if best_score is None or score < best_score:
                best_score = score
                best_choice = (sid, pos)

        if best_choice is None:
            return False, 0, "Optimization failed: unable to place all reservations without overlap."

        target_site, insert_pos = best_choice
        entry = {
            "arrival": arrival,
            "departure": departure,
            "id": res["id"],
            "locked": False,
        }
        schedule[target_site].insert(insert_pos, entry)

        if target_site != res["current_site"]:
            moves.append((target_site, res["id"]))

    if moves:
        conn = get_conn()
        cur = conn.cursor()
        for site_id, res_id in moves:
            cur.execute("UPDATE reservations SET site_id=? WHERE id=?", (site_id, res_id))
        conn.commit()
        conn.close()

    return True, len(moves), None

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

    sites = query_db("SELECT id,name FROM sites ORDER BY CASE WHEN name LIKE 'Site %' THEN CAST(substr(name,6) AS INTEGER) ELSE 9999 END, name")
    reservations = query_db("SELECT site_id, arrival_date, departure_date, status FROM reservations")

    reservations_by_site = {}
    for sid, arrival, departure, status in reservations:
        try:
            arr_d = parse_d(arrival)
            dep_d = parse_d(departure)
        except ValueError:
            continue
        reservations_by_site.setdefault(sid, []).append((arr_d, dep_d, (status or "").lower()))

    for sid in reservations_by_site:
        reservations_by_site[sid].sort(key=lambda x: x[0])

    total_days = len(days)
    site_status_map = {}
    for sid, _sname in sites:
        row = []
        site_res = reservations_by_site.get(sid, [])
        for d in days:
            cell_state = "free"
            for arr_d, dep_d, status in site_res:
                if arr_d <= d < dep_d:
                    cell_state = "confirmed" if status == "confirmed" else "tentative"
                    break
                if arr_d > d:
                    break
            row.append(cell_state)
        site_status_map[sid] = row

    site_lookup = {sid: name for sid, name in sites}

    full_hookup_ids, power_water_ids = [], []
    for sid, name in sites:
        site_num = None
        if name.startswith("Site "):
            try:
                site_num = int(name.split()[1])
            except (ValueError, IndexError):
                site_num = None
        if site_num and 1 <= site_num <= 8:
            full_hookup_ids.append(sid)
        else:
            power_water_ids.append(sid)

    group_definitions = [
        ("Full Hookup", full_hookup_ids, len(full_hookup_ids)),
        ("Power / Water", power_water_ids, 2),
    ]

    def free_run_from(site_id, start_date):
        upcoming = reservations_by_site.get(site_id, [])
        for arr_d, dep_d, _status in upcoming:
            if dep_d <= start_date:
                continue
            if arr_d <= start_date:
                return 0
            gap = (arr_d - start_date).days
            return min(gap, MAX_AVAIL_LOOKAHEAD_DAYS)
        return MAX_AVAIL_LOOKAHEAD_DAYS

    def max_contiguous_free(start_idx, group_site_ids):
        start_date = days[start_idx]
        longest = 0
        for sid in group_site_ids:
            statuses = site_status_map.get(sid, [])
            if start_idx < len(statuses) and statuses[start_idx] == "free":
                run = free_run_from(sid, start_date)
                if run > longest:
                    longest = run
        return longest

    def free_site_count(day_idx, group_site_ids):
        count = 0
        for sid in group_site_ids:
            statuses = site_status_map.get(sid, [])
            if day_idx < len(statuses) and statuses[day_idx] == "free":
                count += 1
        return count

    group_rows = []
    for name, group_site_ids, full_threshold in group_definitions:
        row = []
        for day_index in range(total_days):
            states = [site_status_map.get(sid, ["free"] * total_days)[day_index] for sid in group_site_ids]
            states = [state for state in states if state]
            if not states:
                aggregate = "free"
            elif "free" in states:
                aggregate = "free"
            elif "tentative" in states:
                aggregate = "tentative"
            else:
                aggregate = "confirmed"
            site_count = free_site_count(day_index, group_site_ids)
            actual_run = max_contiguous_free(day_index, group_site_ids) if aggregate == "free" else 0
            display_run = min(actual_run, MAX_AVAIL_DISPLAY_DAYS)
            row.append({
                "state": aggregate,
                "max_run": display_run,
                "actual_run": actual_run,
                "free_sites": site_count,
                "is_full": aggregate == "free" and site_count >= full_threshold,
                "capped": aggregate == "free" and actual_run > MAX_AVAIL_DISPLAY_DAYS
            })
        group_rows.append({
            "name": name,
            "cells": row,
            "threshold": full_threshold
        })

    prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
    next_y, next_m = (year + 1, 1) if month == 12 else (year, month + 1)

    month_name = calendar.month_name[month]

    return render_template(
        "availability.html",
        year=year, month=month, days=days,
        group_rows=group_rows, today=today,
        month_name=month_name,
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

    scope = request.args.get("scope", "month").lower()
    if scope not in {"month", "week"}:
        scope = "month"
    weekly_mode = scope == "week"

    today = date.today()
    month_param = request.args.get("month")
    year_param = request.args.get("year")

    def parse_year_month():
        base_year, base_month = today.year, today.month
        if month_param and "-" in month_param:
            try:
                raw_y, raw_m = month_param.split("-", 1)
                base_year, base_month = int(raw_y), int(raw_m)
            except ValueError:
                pass
        else:
            try:
                base_year = int(year_param)
            except (TypeError, ValueError):
                base_year = today.year
            try:
                base_month = int(request.args.get("month", base_month))
            except (TypeError, ValueError):
                base_month = today.month
        base_month = max(1, min(12, base_month))
        return base_year, base_month

    filter_year, filter_month = parse_year_month()
    month_value = f"{filter_year:04d}-{filter_month:02d}"

    week_options = week_options_for_month(filter_year, filter_month)
    week_start_param = request.args.get("start")
    selected_week_start = ""
    week_range_label = ""
    heading_label = ""
    week_start = None

    if weekly_mode:
        if not week_options:
            fallback_start = date(filter_year, filter_month, 1)
            week_options = [{
                "value": fallback_start.isoformat(),
                "label": f"Week of {calendar.month_name[fallback_start.month]} {fallback_start.day}",
                "range": f"{calendar.month_abbr[fallback_start.month]} {fallback_start.day} – {calendar.month_abbr[(fallback_start + timedelta(days=6)).month]} {(fallback_start + timedelta(days=6)).day}",
            }]
        default_start = week_options[0]["value"]
        week_start_value = week_start_param or default_start
        if not any(opt["value"] == week_start_value for opt in week_options):
            week_start_value = default_start
        try:
            week_start = datetime.strptime(week_start_value, "%Y-%m-%d").date()
        except ValueError:
            week_start = datetime.strptime(default_start, "%Y-%m-%d").date()
            week_start_value = default_start
        selected_week_start = week_start_value
        week_end = week_start + timedelta(days=6)
        if week_start.month == week_end.month and week_start.year == week_end.year:
            week_range_label = f"{calendar.month_name[week_start.month]} {week_start.day} – {week_end.day}"
        else:
            week_range_label = (
                f"{calendar.month_name[week_start.month]} {week_start.day} – "
                f"{calendar.month_name[week_end.month]} {week_end.day}"
            )
        heading_label = f"Week of {calendar.month_name[week_start.month]} {week_start.day}"
        days = [week_start + timedelta(days=i) for i in range(7)]
    else:
        num_days = calendar.monthrange(filter_year, filter_month)[1]
        days = [date(filter_year, filter_month, d) for d in range(1, num_days + 1)]
        heading_label = f"{calendar.month_name[filter_month]} {filter_year}"

    day_iso_list = [d.isoformat() for d in days]
    start_iso, end_iso = day_iso_list[0], day_iso_list[-1]

    sites = query_db(
        "SELECT id,name FROM sites ORDER BY CASE WHEN name LIKE 'Site %' THEN CAST(substr(name,6) AS INTEGER) ELSE 9999 END, name"
    )
    site_display = {sid: short_site_label(name) for sid, name in sites}

    reservations = query_db(
        """
        SELECT r.id, r.site_id, r.guest_name, r.phone, r.email,
               r.arrival_date, r.departure_date, r.status,
               r.rv_size, r.num_adults, r.num_children, r.paid, r.notes,
               r.site_locked
        FROM reservations r
        WHERE NOT (r.departure_date < ? OR r.arrival_date > ?)
        ORDER BY r.site_id ASC, r.arrival_date
    """,
        (start_iso, end_iso),
    )

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
            "notes": r[12] or "",
            "site_locked": bool(r[13]),
        })

    grid = {sid: build_row_cells(days, res_map.get(sid, [])) for sid, _ in sites}

    min_period_start = date(today.year, today.month, 1)
    prev_url = next_url = None
    prev_disabled = False

    if weekly_mode:
        week_start_dt = week_start or days[0]
        prev_start = week_start_dt - timedelta(days=7)
        next_start = week_start_dt + timedelta(days=7)
        if prev_start >= min_period_start:
            prev_url = url_for(
                "dashboard",
                scope="week",
                month=f"{prev_start.year:04d}-{prev_start.month:02d}",
                start=prev_start.isoformat(),
            )
        else:
            prev_disabled = True
        next_url = url_for(
            "dashboard",
            scope="week",
            month=f"{next_start.year:04d}-{next_start.month:02d}",
            start=next_start.isoformat(),
        )
        export_link = url_for(
            "export_reservations",
            scope="week",
            month=month_value,
            start=selected_week_start or day_iso_list[0],
        )
    else:
        def prev_ym(y, m):
            return (y - 1, 12) if m == 1 else (y, m - 1)

        def next_ym(y, m):
            return (y + 1, 1) if m == 12 else (y, m + 1)

        prev_year, prev_month = prev_ym(filter_year, filter_month)
        next_year, next_month = next_ym(filter_year, filter_month)
        prev_start = date(prev_year, prev_month, 1)
        if prev_start >= min_period_start:
            prev_url = url_for("dashboard", scope="month", year=prev_year, month=prev_month)
        else:
            prev_disabled = True
        next_url = url_for("dashboard", scope="month", year=next_year, month=next_month)
        export_link = url_for("export_reservations", scope="month", year=filter_year, month=filter_month)

    return render_template(
        "dashboard.html",
        sites=sites,
        days=days,
        day_iso_list=day_iso_list,
        grid=grid,
        heading_label=heading_label,
        site_display=site_display,
        weekly_mode=weekly_mode,
        week_options=week_options,
        selected_week_start=selected_week_start,
        week_range_label=week_range_label,
        month_value=month_value,
        prev_url=prev_url,
        next_url=next_url,
        prev_disabled=prev_disabled,
        export_link=export_link,
        year=filter_year,
        month=filter_month,
    )


# -----------------------
# CRUD API
# -----------------------
@app.route("/api/reservation/add", methods=["POST"])
def add_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()

    raw_site = (data.get("site_id") or "").strip()
    auto_assign = raw_site.lower() in {"", "auto", "none", "null"}
    if auto_assign:
        site_id = None
    else:
        try:
            site_id = int(raw_site)
        except ValueError:
            return jsonify({"ok": False, "error": "Invalid site selection"})
    start = data["arrival"]
    end = data["departure"]

    status = (data.get("status") or "tentative").lower()
    status = status if status in {"tentative", "confirmed"} else "tentative"
    paid_val = data.get("paid", "no")
    paid = "yes" if str(paid_val).lower() in {"yes", "y", "true", "1"} else "no"
    if status != "confirmed":
        paid = "no"

    warnings = []
    site_locked = 1 if to_bool(data.get("site_locked")) else 0

    if auto_assign:
        assigned_site, auto_warnings, split_required, error_message, _moved = auto_assign_site(start, end)
        warnings.extend(auto_warnings)
        if assigned_site is None:
            if split_required:
                return jsonify({"ok": False, "error": "Requested stay requires splitting across multiple sites.", "split_required": True})
            return jsonify({"ok": False, "error": error_message or "No availability"})
        site_id = assigned_site
    else:
        if not is_available(site_id, start, end):
            ok_opt, moved, error_message = run_optimizer_internal()
            if not ok_opt:
                return jsonify({"ok": False, "error": error_message or "Site not available"})
            if not is_available(site_id, start, end):
                return jsonify({"ok": False, "error": "Site not available for requested dates"})
            if moved:
                site_lookup = {sid: name for sid, name in get_site_rows()}
                warnings.append(f"Optimizer moved {moved} reservation(s) to free {site_lookup.get(site_id, f'Site {site_id}')}.")

    site_rows_lookup = {sid: name for sid, name in get_site_rows()}

    query_db("""
        INSERT INTO reservations
        (site_id, guest_name, phone, email, arrival_date, departure_date,
         status, rv_size, num_campers, paid, notes, num_adults, num_children,
         site_locked)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (site_id, data["guest"], data.get("phone"), data.get("email"),
          start, end, status, data.get("rv_size"),
          data.get("num_campers"), paid,
          data.get("notes",""), data.get("num_adults"), data.get("num_children"),
          site_locked))
    return jsonify({"ok": True, "warnings": warnings, "site_name": site_rows_lookup.get(site_id, f"Site {site_id}")})

@app.route("/api/reservation/update", methods=["POST"])
def update_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()
    res_id = int(data["id"])
    try:
        site_id = int(data["site_id"])
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid site selection"})
    start = data["arrival"]
    end = data["departure"]

    status = (data.get("status") or "tentative").lower()
    status = status if status in {"tentative", "confirmed"} else "tentative"
    paid_val = data.get("paid", "no")
    paid = "yes" if str(paid_val).lower() in {"yes", "y", "true", "1"} else "no"
    if status != "confirmed":
        paid = "no"

    warnings = []

    if not is_available(site_id, start, end, exclude_res_id=res_id):
        ok_opt, moved, error_message = run_optimizer_internal()
        if not ok_opt:
            return jsonify({"ok": False, "error": error_message or "Site not available"})
        if not is_available(site_id, start, end, exclude_res_id=res_id):
            return jsonify({"ok": False, "error": "Site not available for requested dates"})
        if moved:
            site_lookup = {sid: short_site_label(name) for sid, name in get_site_rows()}
            warnings.append(f"Optimizer moved {moved} reservation(s) to free {site_lookup.get(site_id, f'Site {site_id}')}.")

    site_locked = 1 if to_bool(data.get("site_locked")) else 0

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
               num_children=?,
               site_locked=?
         WHERE id=?
    """, (data["guest"], data.get("phone"), data.get("email"),
          start, end, site_id, status,
          data.get("rv_size"), data.get("num_campers"),
          paid, data.get("notes",""),
          data.get("num_adults"), data.get("num_children"),
          site_locked,
          res_id))
    return jsonify({"ok": True, "warnings": warnings})

@app.route("/api/reservation/move", methods=["POST"])
def move_reservation():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not authorized"})
    data = request.get_json()
    res_id = int(data["id"])
    site_id = int(data["site_id"])
    new_start = data["arrival"]
    new_end = data["departure"]

    current = query_db(
        "SELECT site_locked, site_id FROM reservations WHERE id=?",
        (res_id,),
        one=True
    )
    if not current:
        return jsonify({"ok": False, "error": "Reservation not found"})
    locked_flag, current_site = current
    if locked_flag and site_id != current_site:
        return jsonify({"ok": False, "error": "Reservation is locked to its current site"})

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

    ok, moved, error = run_optimizer_internal()
    if not ok:
        return jsonify({"ok": False, "error": error or "Optimization failed"})
    return jsonify({"ok": True, "moved": moved})

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


def week_options_for_month(year: int, month: int):
    """Return week selector metadata for a given month."""
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    options, seen = [], set()
    for week in cal.monthdatescalendar(year, month):
        start_d, end_d = week[0], week[-1]
        if start_d.month != month and end_d.month != month:
            continue
        if start_d in seen:
            continue
        seen.add(start_d)
        start_label = f"Week of {calendar.month_name[start_d.month]} {start_d.day}"
        start_abbr = calendar.month_abbr[start_d.month]
        end_abbr = calendar.month_abbr[end_d.month]
        if start_d.year == end_d.year:
            range_label = f"{start_abbr} {start_d.day} – {end_abbr} {end_d.day}"
        else:
            range_label = (
                f"{start_abbr} {start_d.day} {start_d.year} – "
                f"{end_abbr} {end_d.day} {end_d.year}"
            )
        options.append({
            "value": start_d.isoformat(),
            "label": start_label,
            "range": range_label,
        })
    options.sort(key=lambda opt: opt["value"])
    return options


@app.route("/export")
def export_reservations():
    if "user_id" not in session:
        return redirect(url_for("login"))

    mode = request.args.get("scope", "month")  # "month", "week", or "range"
    today = date.today()

    year_param = request.args.get("year")
    month_param = request.args.get("month")
    try:
        year = int(year_param)
    except (TypeError, ValueError):
        year = today.year
    try:
        month = int(month_param)
    except (TypeError, ValueError):
        month = today.month
    if month_param and "-" in month_param:
        try:
            y_str, m_str = month_param.split("-", 1)
            year = int(y_str)
            month = int(m_str)
        except ValueError:
            pass
    month = max(1, min(12, month))

    from_month = request.args.get("from")  # 'YYYY-MM'
    to_month = request.args.get("to")      # 'YYYY-MM'
    week_start_param = request.args.get("start")

    form_from_value = f"{year:04d}-{month:02d}"
    form_to_value = form_from_value
    selected_week_start = ""

    sites = query_db(
        "SELECT id,name FROM sites ORDER BY CASE WHEN name LIKE 'Site %' THEN CAST(substr(name,6) AS INTEGER) ELSE 9999 END, name"
    )
    site_lookup = {sid: name for sid, name in sites}
    site_short_lookup = {sid: short_site_label(name) for sid, name in sites}
    sections = []

    def collect_period(day_list, heading_label, heading_year, period_type="month", extra=None):
        s_iso, e_iso = day_list[0].isoformat(), day_list[-1].isoformat()
        rows = query_db(
            """
            SELECT r.id, r.site_id, r.guest_name, r.phone, r.email,
                   r.arrival_date, r.departure_date, r.status,
                   r.rv_size, r.num_campers, r.paid, r.notes,
                   r.num_adults, r.num_children, r.site_locked
            FROM reservations r
            WHERE NOT (r.departure_date < ? OR r.arrival_date > ?)
            ORDER BY r.site_id ASC, r.arrival_date
            """,
            (s_iso, e_iso),
        )

        res_map, all_res_list = {}, []
        for r in rows:
            rd = {
                "id": r[0],
                "site_id": r[1],
                "guest": r[2],
                "phone": r[3],
                "email": r[4],
                "arrival": r[5],
                "departure": r[6],
                "status": r[7],
                "rv_size": r[8],
                "num_campers": r[9],
                "paid": r[10],
                "notes": r[11] or "",
                "num_adults": r[12] if r[12] is not None else 0,
                "num_children": r[13] if r[13] is not None else 0,
                "site_locked": bool(r[14]),
            }
            res_map.setdefault(r[1], []).append(rd)
            all_res_list.append(rd)

        all_res_list.sort(key=lambda item: (item["site_id"], item["arrival"], item["departure"], item["id"]))

        grid = {}
        for sid, _sname in sites:
            grid[sid] = build_row_cells(day_list, res_map.get(sid, []))

        entry = {
            "year": heading_year,
            "month": day_list[0].month,
            "month_label": heading_label,
            "days": day_list,
            "grid": grid,
            "all_res_list": all_res_list,
            "period_type": period_type,
        }
        if extra:
            entry.update(extra)
        sections.append(entry)

    def collect_month_data(yy, mm):
        nd = calendar.monthrange(yy, mm)[1]
        dlist = [date(yy, mm, d) for d in range(1, nd + 1)]
        collect_period(dlist, calendar.month_name[mm], str(yy), period_type="month")

    def collect_week_data(start_dt: date):
        days = [start_dt + timedelta(days=i) for i in range(7)]
        end_dt = days[-1]
        if start_dt.year == end_dt.year:
            heading_year = str(start_dt.year)
        else:
            heading_year = f"{start_dt.year} / {end_dt.year}"
        if start_dt.month == end_dt.month:
            heading_label = f"Week of {calendar.month_name[start_dt.month]} {start_dt.day}"
        else:
            heading_label = (
                f"Week of {calendar.month_name[start_dt.month]} {start_dt.day} – "
                f"{calendar.month_name[end_dt.month]} {end_dt.day}"
            )
        extra = {
            "week_start": start_dt,
            "week_end": end_dt,
        }
        collect_period(days, heading_label, heading_year, period_type="week", extra=extra)

    week_options = []

    if mode == "range" and from_month:
        if not to_month:
            to_month = from_month
        try:
            fy, fm = map(int, from_month.split("-"))
            ty, tm = map(int, to_month.split("-"))
        except ValueError:
            fy, fm = year, month
            ty, tm = year, month
        if (fy, fm) > (ty, tm):
            fy, fm, ty, tm = ty, tm, fy, fm
        year, month = fy, fm
        form_from_value = f"{fy:04d}-{fm:02d}"
        form_to_value = f"{ty:04d}-{tm:02d}"
        for (yy, mm) in month_iter(fy, fm, ty, tm):
            collect_month_data(yy, mm)
        week_options = week_options_for_month(year, month)
    elif mode == "week":
        week_options = week_options_for_month(year, month)
        if not week_options:
            fallback_start = date(year, month, 1)
            fallback_end = fallback_start + timedelta(days=6)
            start_abbr = calendar.month_abbr[fallback_start.month]
            end_abbr = calendar.month_abbr[fallback_end.month]
            if fallback_start.year == fallback_end.year:
                range_label = f"{start_abbr} {fallback_start.day} – {end_abbr} {fallback_end.day}"
            else:
                range_label = (
                    f"{start_abbr} {fallback_start.day} {fallback_start.year} – "
                    f"{end_abbr} {fallback_end.day} {fallback_end.year}"
                )
            week_options.append({
                "value": fallback_start.isoformat(),
                "label": f"Week of {calendar.month_name[fallback_start.month]} {fallback_start.day}",
                "range": range_label,
            })
        default_start = week_options[0]["value"]
        week_start_value = week_start_param or default_start
        try:
            week_start_dt = datetime.strptime(week_start_value, "%Y-%m-%d").date()
        except ValueError:
            week_start_dt = datetime.strptime(default_start, "%Y-%m-%d").date()
            week_start_value = week_start_dt.isoformat()
        selected_week_start = week_start_dt.isoformat()
        if all(opt["value"] != selected_week_start for opt in week_options):
            end_dt = week_start_dt + timedelta(days=6)
            start_abbr = calendar.month_abbr[week_start_dt.month]
            end_abbr = calendar.month_abbr[end_dt.month]
            if week_start_dt.year == end_dt.year:
                range_label = f"{start_abbr} {week_start_dt.day} – {end_abbr} {end_dt.day}"
            else:
                range_label = (
                    f"{start_abbr} {week_start_dt.day} {week_start_dt.year} – "
                    f"{end_abbr} {end_dt.day} {end_dt.year}"
                )
            week_options.append({
                "value": selected_week_start,
                "label": f"Week of {calendar.month_name[week_start_dt.month]} {week_start_dt.day}",
                "range": range_label,
            })
            week_options.sort(key=lambda opt: opt["value"])
        collect_week_data(week_start_dt)
        form_from_value = f"{year:04d}-{month:02d}"
        form_to_value = form_from_value
    else:
        if from_month:
            try:
                fy, fm = map(int, from_month.split("-"))
                year, month = fy, fm
            except ValueError:
                pass
        collect_month_data(year, month)
        form_from_value = f"{year:04d}-{month:02d}"
        form_to_value = form_from_value
        week_options = week_options_for_month(year, month)

    month_value = f"{year:04d}-{month:02d}"

    return render_template(
        "export.html",
        scope=mode,
        year=year,
        month=month,
        month_value=month_value,
        sites=sites,
        site_lookup=site_lookup,
        site_short_lookup=site_short_lookup,
        sections=sections,
        form_from_value=form_from_value,
        form_to_value=form_to_value,
        week_options=week_options,
        selected_week_start=selected_week_start,
        weekly_mode=(mode == "week"),
    )

# -----------------------
# boot
# -----------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
