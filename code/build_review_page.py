"""Build a blinded, local-only human review page from a rank_candidates.py run.

    python code/build_review_page.py --run-dir data/matching/<ranking_run> --top 5

Writes into <run-dir>/review/: review.html (open it in a browser; works offline, nothing is uploaded),
review_pool_blinded.json, and answer_key.json (which method/rank returned each pair; do not open
until labelling is done). The page hides method, rank and score and shuffles all pairs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
BOTH = {"ai_support_lead_to_solutions_engineer", "ai_finance_analyst_to_data_scientist"}


def _assignees(profile_id: str, fictional_index: int) -> list[str]:
    if profile_id.startswith("real_jesse"):
        return ["jesse"]
    if profile_id.startswith("real_jacob"):
        return ["jacob"]
    if profile_id in BOTH:
        return ["jesse", "jacob"]
    return ["jesse"] if fictional_index % 2 == 0 else ["jacob"]


def build(run_dir: Path, top: int, profile_paths: list[Path]) -> None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    corpus_path = Path(manifest["corpus_dir"]) / "jobs_prepared_corpus.jsonl"
    rankings = list(csv.DictReader((run_dir / "rankings.csv").open(encoding="utf-8")))
    wanted: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rankings:
        if int(row["rank"]) <= top:
            wanted[(row["profile_id"], row["job_id"])].append({"method": row["method"], "rank": int(row["rank"])})
    job_ids = {job_id for _, job_id in wanted}
    jobs: dict[str, dict[str, Any]] = {}
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["job_id"] in job_ids:
                jobs[record["job_id"]] = record
    profiles: dict[str, dict[str, Any]] = {}
    for path in profile_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in (data["profiles"] if isinstance(data, dict) else data):
            profiles[row["profile_id"]] = row
    fictional_order = {pid: i for i, pid in enumerate(p for p in manifest["profiles"] if not p.startswith("real_"))}

    def card_profile(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "profile_id": row["profile_id"], "goal": row.get("career_transition_goal", ""),
            "target_titles": row.get("desired_roles", []), "experience_level": row.get("experience_level", ""),
            "work_history": row.get("work_history", []), "accomplishments": row.get("accomplishments", []),
            "skills": [f"{s['skill']}: {s['evidence']}" for s in row.get("skills_with_evidence", [])],
            "education": row.get("education_certifications", []),
            "unknown_not_evidenced": (row.get("qualification_disclosures") or {}).get("unknown", []),
            "explicitly_absent": (row.get("qualification_disclosures") or {}).get("explicit_absence", []),
            "location": row.get("location"),
        }

    pairs = sorted(wanted)
    random.Random(SEED).shuffle(pairs)  # order within a candidate is shuffled so method/rank cannot be inferred
    # Group by candidate so a partly finished session still leaves complete candidates. Real no-title profiles
    # first, then fictional, then real profiles that name target titles.
    def group_rank(profile_id: str) -> tuple[int, int]:
        if profile_id.startswith("real_") and profile_id.endswith("_notitle"):
            return (0, sorted(p for p in profiles if p.startswith("real_") and p.endswith("_notitle")).index(profile_id))
        if profile_id.startswith("real_"):
            return (2, sorted(p for p in profiles if p.startswith("real_") and not p.endswith("_notitle")).index(profile_id))
        return (1, fictional_order.get(profile_id, 0))
    pairs.sort(key=lambda pair: group_rank(pair[0]))  # stable sort keeps the shuffled order inside each candidate
    pool, key = [], {}
    for index, (profile_id, job_id) in enumerate(pairs, 1):
        job = jobs[job_id]
        # Stable id from the pair itself: regenerating the page keeps labels for unchanged pairs.
        presentation_id = "R" + hashlib.sha1(f"{profile_id}|{job_id}".encode()).hexdigest()[:10]
        company = job.get("company_name_included")
        pool.append({
            "presentation_id": presentation_id, "profile": card_profile(profiles[profile_id]),
            "job": {"title": job.get("title_raw") or job.get("title_searchable"), "company": company,
                    "location": job.get("location_raw") if isinstance(job.get("location_raw"), str) else json.dumps(job.get("location_raw")),
                    "status_raw": job.get("status_raw"), "seniority": job.get("seniority"),
                    "snapshot_date": job.get("snapshot_date"), "description": job.get("description_text") or ""},
            "assigned_to": _assignees(profile_id, fictional_order.get(profile_id, 0)),
        })
        key[presentation_id] = {"profile_id": profile_id, "job_id": job_id, "returned_by": wanted[(profile_id, job_id)]}
    out = run_dir / "review"
    out.mkdir(exist_ok=False)
    (out / "review_pool_blinded.json").write_text(json.dumps(pool, indent=1), encoding="utf-8")
    (out / "answer_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    page = HTML.replace("__POOL_JSON__", json.dumps(pool).replace("</", "<\\/"))
    (out / "review.html").write_text(page, encoding="utf-8")
    print(f"{len(pool)} pairs for {len(profiles)} profiles -> {out / 'review.html'}")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match Review</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--mute:#5b6570;--line:#d9dee4;--accent:#1f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--card:#1c2025;--ink:#e8ebef;--mute:#9aa4af;--line:#2d333b;--accent:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;z-index:2}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px;max-width:1500px;margin:auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;overflow:auto;max-height:70vh}
h2{margin:0 0 6px;font-size:17px}h3{margin:12px 0 4px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
ul{margin:4px 0;padding-left:20px}pre{white-space:pre-wrap;font:inherit;margin:0}
.meta{color:var(--mute);font-size:13px}
.hl{background:rgba(255,200,0,.28)}
.bar{grid-column:1/-1;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:8px 12px;cursor:pointer}
button.sel{background:var(--accent);color:#fff;border-color:var(--accent)}
textarea{width:100%;min-height:52px;font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px}
details{grid-column:1/-1}input,select{font:inherit;padding:5px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px}
</style></head><body>
<header>
<label>Reviewer <select id="who"><option value="jesse">Jesse</option><option value="jacob">Jacob</option></select></label>
<label>Show <select id="scope"><option value="mine">only my assigned pairs</option><option value="all">all pairs</option></select></label>
<span id="progress" class="meta"></span>
<button id="next-unlabeled">Next unlabeled</button>
<button id="export">Export my labels</button>
</header>
<main>
<details><summary><b>Rubric (read once)</b></summary>
<p><b>Question:</b> if this person searched for jobs, is this listing worth their time to look at? Judge from the profile and the job text only.</p>
<ul>
<li><b>Worth exploring (1)</b>: the duties plausibly build on what the person has done. Any gaps look learnable, or are simply not mentioned.</li>
<li><b>Maybe (2)</b>: a real stretch, or only part of the job fits.</li>
<li><b>Not realistic (3)</b>: wrong kind of work (for example a title collision), or a clearly mandatory requirement the profile plainly does not meet (for example a PhD or 8+ years in a specialty).</li>
<li><b>Can't tell (4)</b>: the listing text is too thin or garbled to judge.</li>
</ul>
<p>Something the profile does not mention is <b>unknown</b>, not a gap. Do not mark a job down only because a skill was not listed. A different job title than the person's targets is fine. Judge the work. Rows are shuffled and you cannot see which method returned them.</p></details>
<div class="card" id="profile"></div>
<div class="card" id="job"></div>
<div class="bar" id="buttons"></div>
<div class="bar" style="display:block"><textarea id="note" placeholder="One line on why (optional but useful)"></textarea></div>
<div class="bar"><button id="prev">&larr; Previous</button><button id="nextbtn">Next &rarr;</button><span class="meta">Keys: 1-4 label, &larr;/&rarr; move. Labels autosave in this browser; use Export to keep a copy.</span></div>
</main>
<script>
const POOL = __POOL_JSON__;
const LABELS = [["worth_exploring","Worth exploring (1)"],["maybe","Maybe (2)"],["not_realistic","Not realistic (3)"],["cant_tell","Can't tell (4)"]];
const $ = id => document.getElementById(id);
let who = "jesse", pos = 0, view = [], store = {};
const keyName = () => "match_review_labels_" + who;
function load(){ try{ store = JSON.parse(localStorage.getItem(keyName())||"{}"); }catch(e){ store = {}; } }
function save(){ try{ localStorage.setItem(keyName(), JSON.stringify(store)); }catch(e){} }
function rebuild(){
  view = POOL.filter(p => $("scope").value==="all" || p.assigned_to.includes(who));
  pos = Math.min(pos, Math.max(0, view.length-1)); render();
}
function list(items){ const ul=document.createElement("ul"); (items||[]).forEach(t=>{const li=document.createElement("li"); li.textContent=t; ul.appendChild(li);}); return ul; }
function section(parent,title,node){ const h=document.createElement("h3"); h.textContent=title; parent.appendChild(h); parent.appendChild(node); }
function txt(s){ const p=document.createElement("p"); p.style.margin="2px 0"; p.textContent=s||"(none)"; return p; }
function highlighted(text){
  const frag=document.createDocumentFragment(), re=/(required|requirements?|must|minimum|preferred|years?|degree|certif\w*|licen[sc]e\w*)/i;
  const pre=document.createElement("pre");
  text.split("\n").forEach((line,i)=>{ if(i) pre.appendChild(document.createTextNode("\n"));
    if(re.test(line)){ const s=document.createElement("span"); s.className="hl"; s.textContent=line; pre.appendChild(s);} else pre.appendChild(document.createTextNode(line)); });
  frag.appendChild(pre); return frag;
}
function render(){
  const total=view.length, done=view.filter(p=>store[p.presentation_id]).length;
  let cand = "";
  if(total){ const pid=view[pos].profile.profile_id, grp=view.filter(p=>p.profile.profile_id===pid);
    cand = ` | this candidate: ${grp.filter(p=>store[p.presentation_id]).length} of ${grp.length} labeled`; }
  $("progress").textContent = total? `Pair ${pos+1} of ${total} | labeled ${done}${cand}` : "No pairs";
  const pr=$("profile"), jb=$("job"); pr.replaceChildren(); jb.replaceChildren();
  if(!total) return;
  const p=view[pos], x=p.profile, j=p.job;
  const h=document.createElement("h2"); h.textContent="Candidate"; pr.appendChild(h);
  const m=document.createElement("div"); m.className="meta"; m.textContent=[x.experience_level,x.location].filter(Boolean).join(" | "); pr.appendChild(m);
  section(pr,"Goal",txt(x.goal));
  if(x.target_titles&&x.target_titles.length) section(pr,"Titles they named",list(x.target_titles)); else section(pr,"Titles they named",txt("none given"));
  section(pr,"Work history",list(x.work_history)); section(pr,"Accomplishments",list(x.accomplishments));
  section(pr,"Skills (with evidence)",list(x.skills)); section(pr,"Education",list(x.education));
  section(pr,"Not mentioned / unknown (not a gap)",list(x.unknown_not_evidenced));
  if(x.explicitly_absent&&x.explicitly_absent.length) section(pr,"Explicitly does not have",list(x.explicitly_absent));
  const t=document.createElement("h2"); t.textContent=j.title||"(untitled)"; jb.appendChild(t);
  const jm=document.createElement("div"); jm.className="meta";
  jm.textContent=[j.company, j.location, j.seniority, "status in snapshot: "+(j.status_raw||"unknown")+" (not verified as open)", "historical listing from "+(j.snapshot_date||"snapshot")].filter(Boolean).join(" | "); jb.appendChild(jm);
  jb.appendChild(highlighted(j.description||"")); jb.scrollTop=0;
  const bar=$("buttons"); bar.replaceChildren(); const cur=store[p.presentation_id];
  LABELS.forEach(([v,l])=>{ const b=document.createElement("button"); b.textContent=l; if(cur&&cur.label===v) b.className="sel"; b.onclick=()=>setLabel(v); bar.appendChild(b); });
  $("note").value = cur? (cur.note||"") : "";
}
function setLabel(v){ const p=view[pos]; store[p.presentation_id]={label:v,note:$("note").value,time:new Date().toISOString()}; save(); render(); }
$("note").addEventListener("change",()=>{ const p=view[pos]; if(p&&store[p.presentation_id]){ store[p.presentation_id].note=$("note").value; save(); } });
function go(d){ if(!view.length) return; pos=(pos+d+view.length)%view.length; render(); }
$("prev").onclick=()=>go(-1); $("nextbtn").onclick=()=>go(1);
$("next-unlabeled").onclick=()=>{ for(let i=1;i<=view.length;i++){ const k=(pos+i)%view.length; if(!store[view[k].presentation_id]){pos=k;break;} } render(); };
$("who").onchange=()=>{ who=$("who").value; load(); pos=0; rebuild(); };
$("scope").onchange=()=>{ pos=0; rebuild(); };
document.addEventListener("keydown",e=>{ if(e.target.tagName==="TEXTAREA"||e.target.tagName==="SELECT") return;
  if(e.key>="1"&&e.key<="4") setLabel(LABELS[+e.key-1][0]); else if(e.key==="ArrowRight") go(1); else if(e.key==="ArrowLeft") go(-1); });
$("export").onclick=()=>{ const blob=new Blob([JSON.stringify({reviewer:who,exported:new Date().toISOString(),labels:store},null,1)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=`labels_${who}.json`; a.click(); };
load(); rebuild();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    build(args.run_dir, args.top, args.profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
