import { renderBeyondArena } from './figures-ba.js';
import { initDemo } from './demo.js';
import { initHyperparameters } from './figures-hp.js';
// Normalize old shared links; the site now has one published edition.
const canonical = new URL(location.href);
if (canonical.searchParams.has('version') || canonical.searchParams.has('v')) {
  canonical.searchParams.delete('version');
  canonical.searchParams.delete('v');
  history.replaceState(null, '', canonical.pathname + canonical.search + canonical.hash);
}
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d = 0) => Number(v).toLocaleString('en-US', {maximumFractionDigits:d});
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const compact = (v) => v >= 1e6 ? `${num(v / 1e6, 1)}M` : v >= 1e3 ? `${num(v / 1e3, 1)}k` : num(v, v < 1 ? 3 : 0);
const C = {green:'#2b7457',orange:'#be794d',purple:'#8070af',gray:'#9ba69e',ink:'#17221d'};
const state = {eloView:'rank',randomK:64,scalingK:200,heatmapK:200};
let data, hp;
const tooltip = $('chart-tooltip');
let tileTipOwner=null;
function tip(html, x, y) {
  tooltip.innerHTML = html; tooltip.hidden = false;
  tooltip.style.left = `${Math.max(8,Math.min(x + 13,innerWidth-tooltip.offsetWidth-8))}px`;
  tooltip.style.top = `${Math.max(8,Math.min(y + 13,innerHeight-tooltip.offsetHeight-8))}px`;
}
function bindTips(el) {
  el.querySelectorAll('[data-tip]').forEach(node => {
    node.addEventListener('pointermove', e => tip(node.dataset.tip,e.clientX,e.clientY));
    node.addEventListener('pointerleave', () => tooltip.hidden = true);
    node.addEventListener('focus', () => {const r=node.getBoundingClientRect();tip(node.dataset.tip,r.x+r.width/2,r.y+r.height/2)});
    node.addEventListener('blur', () => tooltip.hidden = true);
  });
}
function hideTip(){
  tooltip.hidden=true;
  tileTipOwner=null;
  document.querySelectorAll('.heatmap-cell.selected').forEach(b=>{b.classList.remove('selected');b.setAttribute('aria-pressed',false);b.removeAttribute('aria-describedby')});
}
document.addEventListener('keydown',e=>{if(['Escape','PageUp','PageDown'].includes(e.key))hideTip()});
document.addEventListener('pointerdown',e=>{if(!e.target.closest('.heatmap-cell'))hideTip()});
document.addEventListener('wheel',hideTip,{passive:true});
document.addEventListener('touchmove',hideTip,{passive:true});
function segment(id, key, callback) {
  $(id).addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    $(id).querySelectorAll('button').forEach(x=>{const a=x===b;x.classList.toggle('active',a);x.setAttribute('aria-pressed',a)});
    callback(b.dataset[key]);
  });
}
function table(id, headers, rows, caption) {
  $(id).innerHTML=`<table><caption>${esc(caption)}</caption><thead><tr>${headers.map(x=>`<th scope="col">${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${esc(x ?? '—')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
const txt=(x,y,s,extra='')=>`<text x="${x}" y="${y}" ${extra}>${esc(s)}</text>`;
const line=(x1,y1,x2,y2,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
function chart(id, opts={}) {
  const el=$(id),w=el.clientWidth,h=opts.height||el.clientHeight;
  const m={l:opts.left||54,r:opts.right||24,t:27,b:48};
  const [xmin,xmax]=opts.x||[0,1],[ymin,ymax]=opts.y||[0,1];
  const xf=opts.log?Math.log2:x=>x;
  const x=v=>m.l+(xf(v)-xf(xmin))/(xf(xmax)-xf(xmin))*(w-m.l-m.r);
  const y=v=>h-m.b-(v-ymin)/(ymax-ymin)*(h-m.t-m.b);
  let out='';
  (opts.yticks||[0,.2,.4,.6,.8,1]).filter(v=>v>=ymin&&v<=ymax).forEach(v=>{out+=line(m.l,y(v),w-m.r,y(v),'class="grid-line"')+txt(m.l-10,y(v)+4,opts.yformat?opts.yformat(v):v.toFixed(1),'text-anchor="end"')});
  (opts.xticks||[]).forEach(v=>{if(v<xmin||v>xmax)return;out+=line(x(v),m.t,x(v),h-m.b,'class="grid-line"')+txt(x(v),h-m.b+20,opts.xformat?opts.xformat(v):compact(v),'text-anchor="middle"')});
  out+=txt(m.l,12,opts.ylabel||'','class="axis-title"')+txt((m.l+w-m.r)/2,h-4,opts.xlabel||'','class="axis-title" text-anchor="middle"');
  return {el,w,h,m,x,y,get out(){return out},add:s=>out+=s,finish(){el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" aria-hidden="true">${out}</svg>`;bindTips(el)}};
}
function path(rows,x,y,xkey,ykey) {return rows.map((r,i)=>`${i?'L':'M'}${x(r[xkey])},${y(r[ykey])}`).join(' ')}
function point(x,y,color,html,r=4) {return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" class="plot-point"/><circle class="hit" cx="${x}" cy="${y}" r="11" data-tip="${esc(html)}"/>`}
function modelLabel(id){return data.models.find(m=>m.id===id).label}
function panels(id, type, footers = {}) {
  $(id).innerHTML=data.models.map(m=>`<section class="small-multiple ${type}-panel" data-model="${m.id}" aria-label="${esc(m.label)}"><div class="panel-title"><h4>${esc(m.label)}</h4></div><div id="${type}-${m.id}-chart" class="chart panel-chart" role="img" aria-label="${esc(m.label)} ${type==='random'?'random-search comparison':'synthetic versus real scaling'}"></div>${footers[m.id]||''}</section>`).join('');
}
function renderElo() {
  renderBeyondArena(data,state.eloView);
  bindTips($('elo-chart'));
}
function protocol(id,items){$(id).innerHTML=`<ul>${items.map(item=>`<li>${esc(item)}</li>`).join('')}</ul>`}
function renderRandom() {
  const k=state.randomK,band=$('random-band').checked,rows=data.randomSearch.filter(r=>r.k===k);
  const ymax=Math.ceil(Math.max(...rows.flatMap(r=>[r.selected.winrate,r.selectedReal.winrate,...r.draws.map(x=>x.hi)]))*10)/10+.025;
  const footers=Object.fromEntries(data.models.map(m=>{
    const hh=data.headToHead4000.find(x=>x.model===m.id&&x.k===k);
    return[m.id,`<div class="panel-stat"><strong>${pct(hh.winrate,0)}</strong><span>direct win rate<br>${k} selected vs. 4,000 random</span></div>`];
  }));
  panels('random-chart','random',footers);
  rows.forEach(r=>{
    const c=chart(`random-${r.model}-chart`,{left:40,right:15,x:[.8,5200],y:[0,ymax],log:true,xticks:[1,10,100,1000,4000],yticks:[0,.2,.4,.6,.8],xlabel:'Random subset size K′',ylabel:`Win rate vs. random ${k} ↑`});
    c.add(line(c.m.l,c.y(.5),c.w-c.m.r,c.y(.5),'class="reference-line"'));
    if(band){const area=path(r.draws,c.x,c.y,'kprime','hi')+' '+r.draws.slice().reverse().map(x=>`L${c.x(x.kprime)},${c.y(x.lo)}`).join(' ')+' Z';c.add(`<path d="${area}" fill="${C.orange}" opacity=".17"/>`)}
    c.add(line(c.x(k),c.m.t,c.x(k),c.h-c.m.b,'stroke="#a6b0a4" stroke-dasharray="2 4"'));
    c.add(txt(c.x(k)+5,c.m.t+12,'K′ = K','style="font-size:10px;fill:#7c877d"'));
    [[r.selected,C.green,false,'Synthetic-selected'],[r.selectedReal,C.purple,true,'TabRepo-selected']].forEach(([s,col,dash,label])=>{
      c.add(line(c.x(1),c.y(s.winrate),c.x(4000),c.y(s.winrate),`stroke="${col}" stroke-width="2.5" ${dash?'stroke-dasharray="7 4"':''}`));
      c.add(`<rect x="${c.x(1)}" y="${c.y(s.winrate)-6}" width="${c.x(4000)-c.x(1)}" height="12" fill="transparent" data-tip="${esc(`<strong>${esc(modelLabel(r.model))} · ${label}</strong><br>${k} configurations · ${pct(s.winrate,2)} vs. random ${k}<br><small>C = 4,000; N = ${dash?'182 real':'4,000 synthetic'} tasks</small>`)}"/>`);
    });
    c.add(`<path d="${path(r.draws,c.x,c.y,'kprime','mean')}" class="data-line" stroke="${C.orange}"/>`);
    r.draws.forEach(p=>c.add(point(c.x(p.kprime),c.y(p.mean),C.orange,`<strong>${esc(modelLabel(r.model))} · ${num(p.kprime)} random fits</strong><br>${pct(p.mean,2)} mean win rate vs. random ${k}<br><small>Middle 95% of draws: ${pct(p.lo)}–${pct(p.hi)}<br>${p.nDraws} random subsets from 16,000 configurations</small>`,2.6)));
    c.el.dataset.yMax=ymax;c.finish();
  });
  $('random-k-label').textContent=k;
  $('random-note').textContent='1-fold T · 51 datasets · 200 random draws per budget. Shading contains the middle 95% of draws.';
  protocol('random-protocol',[
    `Every line uses the same random-${k} baseline. A gap between the lines is not a direct pairwise win rate.`,
    'Each orange point averages 200 subsets drawn from a bank of 16,000 configurations. The shaded band shows variation across these draws.',
    `We selected ${k} configurations from a pool of 4,000. The green line uses 4,000 synthetic tasks. The purple line uses 182 TabRepo datasets.`,
    'Evaluation uses the first outer train–test split and one inner fold. We choose a configuration by validation and evaluate it on test data.',
    'The percentages under the plots are separate comparisons against 4,000 random configurations. They are rounded results reported in the repository README.'
  ]);
  table('random-table',['Model','Random fits K′','Mean win rate','2.5% draw','97.5% draw'],rows.flatMap(r=>r.draws.map(x=>[modelLabel(r.model),num(x.kprime),pct(x.mean,2),pct(x.lo,2),pct(x.hi,2)])),`All three models against a common random-${k} reference.`);
}
const superscript = value => String(value).split('').map(x=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[+x]).join('');
function renderScaling() {
  const rows=data.budget.filter(r=>r.k===state.scalingK),cells=(data.budgetCells||[]).filter(r=>r.k===state.scalingK),xs=[...rows,...cells].map(r=>r.budget),xmin=Math.min(...xs)/1.35,xmax=Math.max(...xs)*1.3;
  const ticks=[];for(let e=Math.ceil(Math.log2(xmin));e<=Math.log2(xmax);e++)if(e%3===0)ticks.push(2**e);
  panels('scaling-chart','scaling');
  data.models.forEach(m=>{
    const c=chart(`scaling-${m.id}-chart`,{left:42,right:15,x:[xmin,xmax],y:[.48,.7],log:true,xticks:ticks,xformat:v=>`2${superscript(Math.round(Math.log2(v)))}`,yticks:[.5,.55,.6,.65,.7],yformat:v=>v.toFixed(2),xlabel:'Offline budget B = C × N',ylabel:`Win rate vs. random ${state.scalingK} ↑`});
    c.add(line(c.m.l,c.y(.5),c.w-c.m.r,c.y(.5),'class="reference-line"'));
    if($('scaling-cells').checked)cells.filter(r=>r.model===m.id).forEach(r=>c.add(`<circle class="budget-cell" cx="${c.x(r.budget)}" cy="${c.y(r.winrate)}" r="2.2" fill="${r.source==='synthetic'?C.green:C.purple}" opacity=".18"/>`));
    ['synthetic','tabrepo'].forEach(source=>{
      const series=rows.filter(x=>x.model===m.id&&x.source===source).sort((a,b)=>a.budget-b.budget),col=source==='synthetic'?C.green:C.purple;
      c.add(`<path d="${path(series,c.x,c.y,'budget','winrate')}" class="data-line" stroke="${col}"/>`);
      series.forEach(p=>c.add(point(c.x(p.budget),c.y(p.winrate),col,`<strong>${esc(m.label)} · ${source==='synthetic'?'Synthetic':'TabRepo'}</strong><br>${pct(p.winrate,2)} win rate<br>Budget bin: ${num(p.budget)} fits<br>Observed source budget: ${num(p.sourceBudget)}<br><small>Best observed result within this rounded log₂ bin</small>`,3.2)));
    });c.finish();
  });
  $('scaling-note').textContent='Faint dots show evaluated budgets. Lines connect the best result in each budget bin.';
  protocol('scaling-protocol',[
    'The offline budget counts model fits: C configurations trained on N tasks.',
    'Each budget bin spans a factor of two in fit count. We take the best observed result independently within each bin.',
    'Synthetic training uses up to 4,000 tasks and 4,000 configurations. Real-data training uses up to 182 TabRepo datasets and 16,000 configurations.',
    `Both use 1-fold T and a common random-${state.scalingK} baseline. Equal numbers of fits do not necessarily take the same time.`
  ]);
  table('scaling-table',['Model','Source','Budget bin','Actual fits','Win rate'],rows.map(r=>[modelLabel(r.model),r.source,num(r.budget),num(r.sourceBudget),pct(r.winrate,2)]),'Envelope points are selected independently in each budget bin; raw evaluated cells are included in the figure download.');
}
function renderHeatmap() {
  hideTip();
  const rows=data.surfaces.filter(r=>r.k===state.heatmapK),lookup=new Map(rows.map(r=>[`${r.model}/${r.n}/${r.c}`,r])),domains={};
  const low=.5,exponent=1.7;
  const color=(v,high)=>{const t=Math.pow(Math.max(0,Math.min(1,(v-low)/(high-low))),exponent),a=[255,255,255],b=[0,103,61];return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*t)).join(',')})`};
  $('heatmap-chart').innerHTML=data.models.map(m=>{
    const subset=rows.filter(r=>r.model===m.id),ns=[...new Set(subset.map(r=>r.n))].sort((a,b)=>b-a),cs=[...new Set(subset.map(r=>r.c))].sort((a,b)=>a-b),high=Math.max(...subset.map(r=>r.winrate));domains[m.id]={ns,cs};
    const showTick=(i,length)=>i===0||i===length-1||i%Math.ceil((length-1)/4)===0;
    let html=`<section class="small-multiple heatmap-panel" data-model="${m.id}" data-color-min="${low}" data-color-max="${high}" data-color-exponent="${exponent}" aria-label="${esc(m.label)} scaling heatmap"><div class="panel-title"><h4>${esc(m.label)}</h4></div><div class="heatmap-axis">Synthetic tasks N ↑</div><div class="heatmap-grid" style="grid-template-columns:30px repeat(${cs.length},minmax(0,1fr))">`;
    ns.forEach((n,ni)=>{html+=`<span class="row-label">${showTick(ni,ns.length)?compact(n):''}</span>`;cs.forEach(c=>{const r=lookup.get(`${m.id}/${n}/${c}`);html+=r?`<button class="heatmap-cell" style="background:${color(r.winrate,high)}" data-model="${m.id}" data-n="${n}" data-c="${c}" data-winrate="${r.winrate}" tabindex="${n===4000&&c===4000?0:-1}" aria-label="${esc(m.label)}, ${num(n)} tasks, ${num(c)} configurations: ${pct(r.winrate,2)} win rate" aria-pressed="false"></button>`:'<span class="heatmap-empty" aria-label="Not measured"></span>'})});
    html+=`<span></span>${cs.map((c,i)=>`<span class="column-label">${showTick(i,cs.length)?compact(c):''}</span>`).join('')}</div><div class="heatmap-x-label">Configuration pool C →</div>`;
    const stops=Array.from({length:21},(_,i)=>`${color(low+(high-low)*i/20,high)} ${i*5}%`).join(',');
    const ticks=[low,low+(high-low)/2,high];
    html+=`<div class="panel-colorbar"><div class="colorbar-label"><span>Win rate</span><span>Per-panel scale</span></div><div class="colorbar-track" style="background:linear-gradient(90deg,${stops})"></div><div class="colorbar-values">${ticks.map(v=>`<span>${pct(v,1)}</span>`).join('')}</div></div></section>`;
    return html;
  }).join('');
  function inspect(b,event){
    tileTipOwner=b;
    const model=b.dataset.model,n=+b.dataset.n,c=+b.dataset.c,r=lookup.get(`${model}/${n}/${c}`);
    $('heatmap-chart').querySelectorAll('button').forEach(other=>{
      const active=other===b;other.classList.toggle('selected',active);other.setAttribute('aria-pressed',active);
      other.removeAttribute('aria-describedby');
      if(other.dataset.model===model)other.tabIndex=active?0:-1;
    });
    const rect=b.getBoundingClientRect();
    tip(`<strong>${esc(modelLabel(model))}</strong><div class="tile-score">${pct(r.winrate,2)}<span>win rate vs. random ${state.heatmapK}</span></div><div class="tile-meta"><span>Configurations C</span><b>${num(c)}</b><span>Synthetic tasks N</span><b>${num(n)}</b><span>Offline fits</span><b>${num(c*n)}</b></div><small>95% half-width: ±${(r.ci95*100).toFixed(2)} pp.${c===state.heatmapK?'<br>C = K: 50% by construction.':''}</small>`,event?.clientX??rect.right,event?.clientY??rect.top);
    b.setAttribute('aria-describedby','chart-tooltip');
  }
  function dismiss(b){if(tileTipOwner===b){tooltip.hidden=true;tileTipOwner=null}b.classList.remove('selected');b.setAttribute('aria-pressed',false);b.removeAttribute('aria-describedby')}
  $('heatmap-chart').querySelectorAll('button').forEach(b=>{
    b.addEventListener('pointerenter',e=>{if(e.pointerType!=='touch')inspect(b,e)});
    b.addEventListener('pointermove',e=>{if(e.pointerType!=='touch')inspect(b,e)});
    b.addEventListener('pointerleave',e=>{if(e.pointerType!=='touch')dismiss(b)});
    b.addEventListener('focus',()=>inspect(b));
    b.addEventListener('blur',()=>dismiss(b));
    b.addEventListener('click',e=>inspect(b,e.detail?e:undefined));
    b.addEventListener('keydown',e=>{const moves={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]};if(!moves[e.key])return;e.preventDefault();const model=b.dataset.model,{ns,cs}=domains[model],[dn,dc]=moves[e.key],ni=ns.indexOf(+b.dataset.n)+dn,ci=cs.indexOf(+b.dataset.c)+dc;if(ni<0||ci<0||ni>=ns.length||ci>=cs.length)return;const next=$('heatmap-chart').querySelector(`button[data-model="${model}"][data-n="${ns[ni]}"][data-c="${cs[ci]}"]`);if(next){next.focus();next.click()}});
  });
  $('heatmap-note').textContent=`1-fold T · K = ${state.heatmapK}. Hover or tap for exact values. Each model has its own color scale.`;
  protocol('heatmap-protocol',[
    `Each tile compares a selected portfolio of ${state.heatmapK} configurations with random ${state.heatmapK} on 51 TabArena datasets, using 1-fold T.`,
    'Each color scale runs from 50% to that panel’s maximum. Use the numbers to compare models.',
    'The scale emphasizes larger gains: the normalized gain above 50% is raised to the power 1.7.',
    'When C = K, the portfolio contains the whole candidate pool. Its win rate is 50% by construction.',
    'Only measured settings appear in the grid. Equal spacing between tiles does not mean equal numerical steps. Use arrow keys to move between tiles.'
  ]);
  table('heatmap-table',['Model','Tasks N','Configurations C','Offline fits','Win rate','95% half-width'],rows.map(r=>[modelLabel(r.model),num(r.n),num(r.c),num(r.n*r.c),pct(r.winrate,2),pct(r.ci95,2)]),'All measured cells for the current K, across the three models.');
}
function download(name,rows){
  const text=JSON.stringify(rows,null,2),url=URL.createObjectURL(new Blob([text],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`priortune-${name}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function wireData(){
  segment('elo-view','view',v=>{state.eloView=v;renderElo()});$('elo-regime').addEventListener('change',renderElo);
  [['random','randomK',renderRandom],['scaling','scalingK',renderScaling],['heatmap','heatmapK',renderHeatmap]].forEach(([id,key,render])=>{segment(`${id}-k`,'k',v=>{state[key]=+v;render()})});
  $('random-band').addEventListener('change',renderRandom);$('scaling-cells').addEventListener('change',renderScaling);
  document.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>{
    const type=b.dataset.download,rows=type==='elo'?{elo:data.elo,latency:data.latency}:type==='random'?data.randomSearch.filter(r=>r.k===state.randomK):type==='scaling'?{envelope:data.budget.filter(r=>r.k===state.scalingK),cells:data.budgetCells.filter(r=>r.k===state.scalingK)}:type==='heatmap'?data.surfaces.filter(r=>r.k===state.heatmapK):hp.selected();
    download(type,{provenance:data.meta,results:rows});
  }));
  $('provenance-content').innerHTML=`<p>The measured figures use public PriorTune results from <a href="https://github.com/yandex-research/PriorTune/tree/${data.meta.commit}" target="_blank" rel="noopener">GitHub (${data.meta.commit.slice(0,8)})</a> and <a href="https://huggingface.co/datasets/mrsergazinov/PriorTune/tree/${data.meta.hfCommit}" target="_blank" rel="noopener">Hugging Face (${data.meta.hfCommit.slice(0,8)})</a>. The website does not run experiments.</p><ul><li>Download <a href="data/research.json" download>all figure data</a>, <a href="data/method.json" download>methods and protocols</a>, or the <a href="data/SOURCE-LICENSE.txt" download>source license</a>. Each figure’s Data button exports the current selection.</li><li>The opening illustration follows the trend in <a href="#random-search">Plot 2</a>. Its scores are illustrative.</li><li>BeyondArena uses eight inner folds. The TabArena T ablations here use one, as confirmed by the authors.</li><li>Source-file hashes are included in the JSON. The website repository contains scripts to reproduce the extracted data.</li></ul>`;
  renderElo();renderRandom();renderScaling();renderHeatmap();hp=initHyperparameters(data,{chart,txt,line,path,table,esc,num,pct,tip,tooltip});
  let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>{tooltip.hidden=true;renderElo();renderRandom();renderScaling();hp.render()},150)});
  document.documentElement.dataset.ready='true';
}
async function loadData(){try{const response=await fetch('data/research.json');if(!response.ok)throw Error(response.status);data=await response.json();$('data-error').hidden=true;wireData()}catch(error){$('data-error').hidden=false;console.error('Figure data failed to load',error)}}
$('retry').addEventListener('click',loadData);
$('copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('citation').textContent);$('toast').textContent='BibTeX copied';$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,2400)}catch{$('citation').closest('pre').focus();$('toast').textContent='Select and copy the citation below.';$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,3000)}});

initDemo();loadData();
