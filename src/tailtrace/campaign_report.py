"""Self-contained interactive oracle audit with no network dependencies."""

from __future__ import annotations

import json
from pathlib import Path


def write_campaign_html(data, path):
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    document = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TailTrace · Scheduling oracle audit</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101821;color:#e5edf5;font:16px system-ui;max-width:1400px;padding:36px;margin:auto}h1{font-size:38px;margin:12px 0}h2{font-size:21px}p,small{color:#afbdcc;line-height:1.6}.badge{color:#f4c96d;letter-spacing:.1em;font-size:12px;text-transform:uppercase}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}.card,section{background:#182532;border:1px solid #304457;border-radius:12px;padding:20px}.card strong{display:block;font-size:28px;margin-top:10px}.layout{display:grid;grid-template-columns:1fr 1.3fr;gap:18px}select,input{background:#101821;color:#e5edf5;border:1px solid #526b80;border-radius:6px;padding:9px;max-width:100%;margin:4px 4px 12px 0}#case{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #304457;padding:9px}svg{width:100%;display:block}pre{white-space:pre-wrap;font-size:12px;max-height:340px;overflow:auto}li{margin:10px 0;color:#afbdcc}.legend{display:flex;gap:16px;font-size:12px}.good{color:#75d8ac}.warn{color:#f4c96d}button{background:#263e52;color:#fff;border:1px solid #526b80;padding:9px;border-radius:6px;cursor:pointer}@media(max-width:900px){.layout{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}body{padding:18px}h1{font-size:30px}}
</style></head><body>
<div class="badge">Analytic proxy · exhaustive small-batch audit</div>
<h1>Where does the scheduler fail?</h1>
<p>Inspect every assignment, constraint, and oracle result. A proxy advantage is a hypothesis for a hardware experiment.</p>
<div class="cards" id="cards"></div>
<div class="layout"><section><h2>Proven regret by case</h2><p>Random baseline regret on the horizontal axis; fleet beam-search regret on the vertical axis. Select a dot to inspect its assignment.</p>
<label>Distribution <select id="distribution"><option value="">All</option></select></label>
<label>Fleet <select id="fleet"><option value="">All</option></select></label>
<svg id="plot" viewBox="0 0 560 340" role="img" aria-label="Random versus fleet proxy regret"></svg>
<p id="coverage"></p><div class="legend"><span class="good">● Matches proxy optimum</span><span class="warn">● Positive proven regret</span></div>
</section><section><h2>Assignment inspector</h2><label for="case">Case</label><select id="case"></select>
<p id="status"></p><table><thead><tr><th>Method</th><th>Proxy makespan</th><th>Proven regret</th></tr></thead><tbody id="methods"></tbody></table>
<label>Layout <select id="layout"><option value="fleet_beam">Fleet beam + local search</option><option value="fleet_local">Fleet local search</option><option value="fleet_greedy">Fleet greedy</option><option value="balanced">Legacy balanced</option><option value="random">Random baseline</option><option value="oracle">Oracle incumbent</option></select></label>
<svg id="assignment" viewBox="0 0 660 250" role="img" aria-label="Samples assigned to each physical rank"></svg>
<p>Bars show input sequence lengths. Rank costs use padded width (length − 1). Caps and coefficients remain attached to physical ranks.</p>
<details><summary>Full selected-case evidence</summary><pre id="raw"></pre></details></section></div>
<h2>Measurement limits</h2><ul id="limits"></ul>
<p>Offline artifact · deterministic inputs · no external scripts or uploaded telemetry</p>
<script>
const campaign=__PAYLOAD__;
const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
function node(tag,text,parent){const e=document.createElement(tag);if(text!==null)e.textContent=text;parent.appendChild(e);return e;}
function svg(tag,attrs,parent,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;parent.appendChild(e);return e;}
const percent=x=>x===undefined?'—':(100*x).toFixed(2)+'%';
const summary=campaign.summary;
for(const [label,value] of [['Cases retained',summary.cases],['Proven optimal searches',summary.oracle_status.optimal],['Beam reaches optimum',summary.methods.fleet_beam.optimum_reached],['Worst proven beam regret',percent(summary.methods.fleet_beam.worst_proven_regret??undefined)]]){const card=node('div',null,$('cards'));card.className='card';node('small',label,card);node('strong',String(value),card);}
for(const limit of campaign.limitations)node('li',limit,$('limits'));
for(const key of ['distribution','fleet'])for(const value of [...new Set(campaign.cases.map(c=>c[key]))].sort()){const option=node('option',value,$(key));option.value=value;}
let visible=[];
function inspect(){const c=campaign.cases.find(c=>c.id===$('case').value);if(!c)return;$('status').textContent=`Oracle: ${c.oracle.status} · ${c.oracle.nodes.toLocaleString()} / ${c.oracle.node_budget.toLocaleString()} search nodes · ${c.lengths.length} samples · ${c.models.length} ranks`;$('methods').replaceChildren();for(const [name,result] of Object.entries(c.schedules)){const row=node('tr',null,$('methods'));node('td',name,row);node('td',result.certificate?result.certificate.makespan.toFixed(2):'unavailable',row);node('td',percent(result.proven_regret),row);}
$('raw').textContent=JSON.stringify(c,null,2);const choice=$('layout').value;const groups=choice==='oracle'?c.oracle.groups:c.schedules[choice].groups;const chart=$('assignment');chart.replaceChildren();if(!groups){svg('text',{x:20,y:40,fill:'#f4c96d'},chart,'No feasible assignment recorded');return;}const max=Math.max(...c.lengths);groups.forEach((g,r)=>{const y=30+r*70;svg('text',{x:8,y:y+14,fill:'#e5edf5'},chart,`Rank ${r}`);let x=85;for(const i of g){const barWidth=Math.max(24,450/c.lengths.length);const height=45*c.lengths[i]/max;const rect=svg('rect',{x,y:y+50-height,width:barWidth-4,height,rx:3,fill:r%2?'#7bafe2':'#75d8ac'},chart);svg('title',{},rect,`sample ${i}: ${c.lengths[i]} tokens`);svg('text',{x:x+2,y:y+64,fill:'#afbdcc','font-size':10},chart,String(i));x+=barWidth;}const m=c.models[r];svg('text',{x:420,y:y+14,fill:'#afbdcc','font-size':11},chart,`a=${m.quadratic}, b=${m.linear}, c=${m.overhead}`);svg('text',{x:420,y:y+31,fill:'#afbdcc','font-size':11},chart,`sample cap ${m.max_samples}; padded cap ${m.max_padded_tokens??'none'}`);});}
function render(){visible=campaign.cases.filter(c=>(!$('distribution').value||c.distribution===$('distribution').value)&&(!$('fleet').value||c.fleet===$('fleet').value));$('case').replaceChildren();for(const c of visible){const option=node('option',c.id,$('case'));option.value=c.id;}const points=visible.filter(c=>c.schedules.random.proven_regret!==undefined&&c.schedules.fleet_beam.proven_regret!==undefined);const xmax=Math.max(.05,...points.map(c=>c.schedules.random.proven_regret)),ymax=Math.max(.05,...points.map(c=>c.schedules.fleet_beam.proven_regret));const chart=$('plot');chart.replaceChildren();svg('line',{x1:50,y1:290,x2:530,y2:290,stroke:'#526b80'},chart);svg('line',{x1:50,y1:290,x2:50,y2:20,stroke:'#526b80'},chart);for(let i=0;i<=4;i++){const x=50+480*i/4,y=290-270*i/4;svg('text',{x,y:310,fill:'#afbdcc','font-size':11,'text-anchor':'middle'},chart,percent(xmax*i/4));svg('text',{x:44,y:y+4,fill:'#afbdcc','font-size':11,'text-anchor':'end'},chart,percent(ymax*i/4));}svg('text',{x:260,y:332,fill:'#afbdcc','font-size':12},chart,'Random baseline regret');for(const c of points){const regret=c.schedules.fleet_beam.proven_regret;const dot=svg('circle',{cx:50+480*c.schedules.random.proven_regret/xmax,cy:290-270*regret/ymax,r:5,fill:regret<1e-12?'#75d8ac':'#f4c96d',opacity:.7,tabindex:0,role:'button','aria-label':c.id},chart);dot.style.cursor='pointer';svg('title',{},dot,c.id+' · fleet regret '+percent(regret));const choose=()=>{$('case').value=c.id;inspect();};dot.addEventListener('click',choose);dot.addEventListener('keydown',e=>{if(e.key==='Enter')choose();});}$('coverage').textContent=`${points.length} plotted / ${visible.length} selected cases. Unproven or infeasible comparisons remain in the inspector.`;inspect();}
for(const key of ['distribution','fleet'])$(key).addEventListener('change',render);$('case').addEventListener('change',inspect);$('layout').addEventListener('change',inspect);render();
</script></body></html>"""
    Path(path).write_text(document.replace("__PAYLOAD__", payload))
