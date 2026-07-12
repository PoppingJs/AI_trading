from __future__ import annotations


COMMON_CSS = r"""
:root {
  color-scheme: light;
  font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
  --navy: #111827;
  --page: #f6f7f9;
  --surface: #ffffff;
  --border: #dfe3e8;
  --line: #e8ebef;
  --text: #171717;
  --muted: #667085;
  --blue: #2563eb;
  --blue-soft: #eff6ff;
  --green: #059669;
  --red: #dc2626;
  --amber: #d97706;
  --radius: 7px;
  --shadow: 0 1px 2px rgba(15, 23, 42, .05);
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; background: var(--page); color: var(--text); }
body { font-size: 13px; }
.app-header {
  min-height: 58px; padding: 8px 16px; background: var(--navy); color: white;
  display: flex; align-items: center; justify-content: space-between; gap: 20px;
}
.brand h1 { margin: 0; font-size: 20px; line-height: 1.2; letter-spacing: .01em; }
.brand p { margin: 4px 0 0; color: #cbd5e1; font-size: 12px; }
.system-status { color: #94a3b8; font-size: 12px; white-space: nowrap; }
.top-nav {
  height: 40px; padding: 0 16px; display: flex; align-items: stretch; gap: 20px;
  background: white; border-bottom: 1px solid var(--border);
}
.top-nav a {
  min-width: 70px; padding: 0 4px; color: #344054; text-decoration: none;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; border-bottom: 2px solid transparent;
}
.top-nav a.active { color: var(--blue); border-color: var(--blue); }
.page { padding: 10px 12px 18px; min-width: 0; }
.panel { background: white; border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); }
.toolbar { padding: 10px 12px; }
.toolbar-grid {
  display: grid; grid-template-columns: 1.05fr 1.35fr .85fr .8fr .9fr .85fr auto;
  gap: 12px; align-items: end;
}
.review-toolbar-grid { grid-template-columns: 1.35fr repeat(5, minmax(0, 1fr)) auto; }
.field { min-width: 0; }
.field label { display: block; margin-bottom: 5px; color: #596579; font-size: 11px; }
input, select, button { font: inherit; }
input, select {
  width: 100%; height: 34px; padding: 0 10px; color: #111827; background: white;
  border: 1px solid #d6dbe2; border-radius: 5px; outline: none;
}
input:focus, select:focus { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(37, 99, 235, .1); }
button {
  height: 34px; padding: 0 18px; border: 0; border-radius: 5px; cursor: pointer;
  font-weight: 700; white-space: nowrap;
}
button.primary { color: white; background: var(--blue); }
button.secondary { color: #344054; background: #e9edf2; }
button.danger { color: white; background: var(--red); }
button:disabled { opacity: .5; cursor: not-allowed; }
.safety-note {
  margin-top: 9px; padding-top: 8px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 11px; display: flex; justify-content: space-between; gap: 16px;
}
.job-strip {
  margin-top: 10px; min-height: 46px; padding: 9px 12px; display: flex;
  align-items: center; gap: 14px;
}
.job-name { min-width: 210px; font-weight: 700; }
.progress-track { height: 7px; flex: 1; min-width: 120px; background: #e5e7eb; border-radius: 99px; overflow: hidden; }
.progress-fill { width: 0; height: 100%; background: var(--blue); transition: width .25s ease; }
.job-meta { color: var(--muted); font-size: 11px; white-space: nowrap; }
.metrics { margin-top: 10px; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 9px; }
.metric { min-height: 72px; padding: 10px 12px; }
.metric label { display: block; color: var(--muted); font-size: 11px; }
.metric strong { display: block; margin-top: 10px; font-size: 18px; line-height: 1; }
.metric small { display: block; margin-top: 5px; color: var(--muted); }
.positive { color: var(--green) !important; }
.negative { color: var(--red) !important; }
.charts { margin-top: 10px; display: grid; grid-template-columns: 1.35fr .95fr .95fr; gap: 9px; }
.chart-panel { min-height: 238px; padding: 10px 12px; overflow: hidden; }
.panel-title { margin: 0 0 8px; font-size: 14px; }
.chart-svg { width: 100%; height: 190px; display: block; }
.chart-empty { height: 190px; display: grid; place-items: center; color: #98a2b3; }
.axis-line { stroke: #e5e7eb; stroke-width: 1; }
.tab-row {
  margin-top: 10px; height: 40px; padding: 0 8px; display: flex; align-items: stretch;
  gap: 8px; background: white; border: 1px solid var(--border); border-radius: var(--radius) var(--radius) 0 0;
}
.tab-row button {
  height: 39px; padding: 0 12px; color: #475467; background: transparent;
  border-radius: 0; border-bottom: 2px solid transparent;
}
.tab-row button.active { color: var(--blue); border-color: var(--blue); }
.result-layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 9px; }
.result-layout > * { min-width: 0; }
.result-main, .diagnostics { border-top: 0; border-radius: 0 0 var(--radius) var(--radius); }
.table-wrap { width: 100%; max-width: 100%; min-width: 0; overflow: auto; max-height: 430px; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { position: sticky; top: 0; z-index: 1; padding: 8px; color: #596579; background: #f8fafc; text-align: left; white-space: nowrap; }
td { padding: 7px 8px; border-top: 1px solid var(--line); white-space: nowrap; }
tbody tr:hover { background: #f8fbff; }
.detail-row td { padding: 0; background: #f8fbff; }
.trade-detail { margin: 8px; padding: 10px; border: 1px solid #bfd4ff; border-radius: 5px; display: grid; grid-template-columns: 1fr 1.7fr 1fr; gap: 14px; }
.timeline { display: grid; gap: 5px; }
.timeline div { display: grid; grid-template-columns: 120px 80px 1fr; gap: 8px; }
.diagnostics { padding: 12px; }
.diagnostics h3 { margin: 0 0 10px; font-size: 14px; }
.diagnostic { padding: 9px 0; border-top: 1px solid var(--line); }
.diagnostic:first-of-type { border-top: 0; }
.diagnostic strong { display: block; font-size: 12px; }
.diagnostic p { margin: 4px 0 0; color: var(--muted); font-size: 11px; line-height: 1.5; }
.bar-list { display: grid; gap: 11px; padding-top: 6px; }
.bar-row { display: grid; grid-template-columns: 90px 1fr 52px; align-items: center; gap: 8px; }
.bar-track { height: 10px; background: #edf0f3; position: relative; overflow: hidden; }
.bar-value { height: 100%; background: var(--green); }
.bar-value.loss { background: var(--red); }
.empty-state { min-height: 300px; display: grid; place-items: center; color: #98a2b3; text-align: center; }
.hidden { display: none !important; }
.error-text { color: var(--red); }
@media (max-width: 1280px) {
  .toolbar-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .review-toolbar-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .charts { grid-template-columns: 1fr 1fr; }
  .charts .chart-panel:first-child { grid-column: 1 / -1; }
  .result-layout { grid-template-columns: 1fr; }
  .diagnostics { border-radius: var(--radius); border-top: 1px solid var(--border); }
}
@media (max-width: 720px) {
  .app-header { align-items: flex-start; padding: 9px 12px; }
  .system-status { display: none; }
  .top-nav { gap: 4px; padding: 0 8px; }
  .top-nav a { flex: 1; min-width: 0; }
  .page { padding: 8px; }
  .toolbar-grid, .metrics, .charts { grid-template-columns: 1fr 1fr; }
  #dateField { grid-column: 1 / -1; }
  .review-toolbar-grid { grid-template-columns: 1fr 1fr; }
  .review-toolbar-grid > .field:first-child { grid-column: 1 / -1; }
  .toolbar-grid .run-field { grid-column: 1 / -1; }
  .toolbar-grid .run-field button { width: 100%; }
  .metrics .metric { min-height: 66px; }
  .chart-panel, .charts .chart-panel:first-child { grid-column: 1 / -1; }
  .job-strip { align-items: flex-start; flex-wrap: wrap; }
  .job-name { min-width: 100%; }
  .progress-track { flex-basis: 70%; }
  .tab-row { overflow-x: auto; }
  .tab-row button { flex: 0 0 auto; }
  .trade-detail { grid-template-columns: 1fr; }
}
"""


COMMON_SCRIPT = r"""
const apiToken = localStorage.getItem('AI_TRADING_API_TOKEN');
async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (apiToken) headers['X-API-Token'] = apiToken;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}
function fmtMoney(value, digits = 2) {
  const number = Number(value || 0);
  return `${number >= 0 ? '+' : ''}${number.toFixed(digits)} U`;
}
function fmtPct(value, digits = 2) { return `${(Number(value || 0) * 100).toFixed(digits)}%`; }
function fmtR(value) { return `${Number(value || 0) >= 0 ? '+' : ''}${Number(value || 0).toFixed(2)}R`; }
function fmtDuration(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours >= 24 ? `${Math.floor(hours / 24)}天${hours % 24}小时` : `${hours}h ${minutes}m`;
}
function cls(value) { return Number(value || 0) >= 0 ? 'positive' : 'negative'; }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}
function lineChart(id, points, valueKey, color = '#059669', zero = false) {
  const host = document.getElementById(id);
  if (!host) return;
  if (!points || points.length < 2) { host.innerHTML = '<div class="chart-empty">暂无可绘制数据</div>'; return; }
  const width = 800, height = 190, left = 46, right = 10, top = 12, bottom = 24;
  const values = points.map(point => Number(point[valueKey] || 0));
  let min = Math.min(...values), max = Math.max(...values);
  if (zero) { min = Math.min(min, 0); max = Math.max(max, 0); }
  if (max === min) { max += 1; min -= 1; }
  const x = index => left + index * (width - left - right) / Math.max(points.length - 1, 1);
  const y = value => top + (max - value) * (height - top - bottom) / (max - min);
  const polyline = values.map((value, index) => `${x(index)},${y(value)}`).join(' ');
  const grids = [0, .5, 1].map(ratio => {
    const value = max - (max - min) * ratio; const yy = y(value);
    return `<line class="axis-line" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text x="4" y="${yy+4}" fill="#667085" font-size="10">${value.toFixed(1)}</text>`;
  }).join('');
  host.innerHTML = `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${grids}<polyline points="${polyline}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
}
function barList(id, items, valueKey = 'pnl') {
  const host = document.getElementById(id);
  if (!host) return;
  if (!items || !items.length) { host.innerHTML = '<div class="chart-empty">暂无分类数据</div>'; return; }
  const max = Math.max(...items.map(item => Math.abs(Number(item[valueKey] || 0))), 1);
  host.innerHTML = `<div class="bar-list">${items.slice(0, 8).map(item => {
    const value = Number(item[valueKey] || 0);
    return `<div class="bar-row"><span title="${escapeHtml(item.name || item.reason)}">${escapeHtml(item.name || item.reason)}</span><div class="bar-track"><div class="bar-value ${value < 0 ? 'loss' : ''}" style="width:${Math.abs(value)/max*100}%"></div></div><strong class="${cls(value)}">${valueKey === 'average_r' ? fmtR(value) : value.toFixed(1)}</strong></div>`;
  }).join('')}</div>`;
}
async function refreshHeaderStatus() {
  try {
    const health = await api('/api/health');
    const status = document.getElementById('globalStatus');
    if (status) status.textContent = `${health.running ? '行情与持仓管理运行中' : '服务已就绪'} | ${health.auto_trade ? '自动交易开启' : '自动交易关闭'} | ${new Date().toLocaleString()}`;
  } catch (_) {}
}
refreshHeaderStatus();
setInterval(refreshHeaderStatus, 15000);
"""


def backtest_page() -> str:
    return _page_shell(
        title="历史回测",
        active="backtest",
        body=BACKTEST_BODY,
        script=COMMON_SCRIPT + BACKTEST_SCRIPT,
    )


def review_page() -> str:
    return _page_shell(
        title="交易复盘",
        active="review",
        body=REVIEW_BODY,
        script=COMMON_SCRIPT + REVIEW_SCRIPT,
    )


def _page_shell(*, title: str, active: str, body: str, script: str) -> str:
    nav = {
        "realtime": ("/", "实时交易"),
        "backtest": ("/backtest", "历史回测"),
        "review": ("/review", "交易复盘"),
    }
    links = "".join(
        f'<a href="{path}" class="{"active" if key == active else ""}">{label}</a>'
        for key, (path, label) in nav.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} - AI量化交易平台</title><style>{COMMON_CSS}</style></head>
<body><header class="app-header"><div class="brand"><h1>AI量化交易平台</h1><p>Binance USDT-M 实时行情，本地模拟账户，不会产生真实订单。</p></div><div class="system-status" id="globalStatus">正在读取系统状态...</div></header>
<nav class="top-nav" aria-label="主导航">{links}</nav>{body}<script>{script}</script></body></html>"""


BACKTEST_BODY = r"""
<main class="page">
  <section class="panel toolbar">
    <div class="toolbar-grid">
      <div class="field"><label for="dataSource">数据来源</label><select id="dataSource" onchange="toggleSource()"><option value="demo">演示数据（功能验证）</option><option value="binance">Binance历史K线</option><option value="local">本地数据集</option></select></div>
      <div class="field" id="dateField"><label>回测时间</label><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px"><input id="startDate" type="date"><input id="endDate" type="date"></div></div>
      <div class="field" id="symbolField"><label for="symbol">币种</label><input id="symbol" value="BTCUSDT"></div>
      <div class="field hidden" id="datasetField"><label for="dataset">本地数据集</label><select id="dataset"></select></div>
      <div class="field"><label for="startingEquity">初始资金（USDT）</label><input id="startingEquity" type="number" min="100" value="10000"></div>
      <div class="field"><label for="mode">模式</label><select id="mode"><option value="production">生产一致性</option><option value="legacy">Legacy旧基准</option></select></div>
      <div class="field"><label for="baseTimeframe">基础周期</label><select id="baseTimeframe"><option value="15m">15m</option><option value="1h">1h</option></select></div>
      <div class="field run-field"><button class="primary" id="runButton" onclick="runBacktest()">运行回测</button></div>
    </div>
    <div class="safety-note"><span>只读历史回放，不会启动策略、不会修改当前模拟账户、不会产生真实订单。</span><span id="sourceHint">演示数据仅用于确认功能与排版。</span></div>
  </section>
  <section class="panel job-strip" id="jobStrip">
    <div class="job-name" id="jobName">尚未创建回测任务</div><div class="progress-track"><div class="progress-fill" id="progressFill"></div></div><div class="job-meta" id="jobMeta">等待运行</div><button class="secondary" id="cancelButton" onclick="cancelJob()" disabled>取消</button>
  </section>
  <section class="metrics" id="backtestMetrics"></section>
  <section class="charts">
    <article class="panel chart-panel"><h2 class="panel-title">净值曲线</h2><div id="equityChart"></div></article>
    <article class="panel chart-panel"><h2 class="panel-title">回撤曲线</h2><div id="drawdownChart"></div></article>
    <article class="panel chart-panel"><h2 class="panel-title">累计已实现盈亏</h2><div id="pnlChart"></div></article>
  </section>
  <section class="tab-row" id="backtestTabs"></section>
  <section class="result-layout"><div class="panel result-main" id="backtestResult"></div><aside class="panel diagnostics" id="backtestDiagnostics"><h3>诊断摘要</h3><div class="empty-state">运行回测后生成诊断</div></aside></section>
</main>
"""


BACKTEST_SCRIPT = r"""
let currentJobId = null, pollTimer = null, backtestResult = null, activeBacktestTab = '总体表现';
const backtestTabs = ['总体表现','交易明细','Setup分析','止盈止损','插针与扫损','风险闸门','生产 vs Legacy'];
function initDates() { const end = new Date(); const start = new Date(end); start.setDate(start.getDate()-30); document.getElementById('endDate').value=end.toISOString().slice(0,10); document.getElementById('startDate').value=start.toISOString().slice(0,10); }
async function loadDatasets() { try { const data=await api('/api/backtests/datasets'); const select=document.getElementById('dataset'); select.innerHTML=(data.datasets||[]).map(item=>`<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}${item.has_derivatives?' · 含衍生品':''}</option>`).join('') || '<option value="">暂无本地数据集</option>'; } catch (_) {} }
function toggleSource() { const source=document.getElementById('dataSource').value; document.getElementById('datasetField').classList.toggle('hidden',source!=='local'); document.getElementById('dateField').classList.toggle('hidden',source==='local'); document.getElementById('sourceHint').textContent=source==='demo'?'演示数据仅用于确认功能与排版。':source==='binance'?'在线下载最多90天；无历史衍生品时按价格数据回放。':'从 data/backtests 读取CSV数据。'; }
function renderBacktestTabs(){ document.getElementById('backtestTabs').innerHTML=backtestTabs.map(tab=>`<button class="${tab===activeBacktestTab?'active':''}" onclick="selectBacktestTab('${tab}')">${tab}</button>`).join(''); }
function selectBacktestTab(tab){ activeBacktestTab=tab; renderBacktestTabs(); renderBacktestResult(); }
async function runBacktest(){
  const source=document.getElementById('dataSource').value; const button=document.getElementById('runButton'); button.disabled=true;
  try { const request={data_source:source,dataset:document.getElementById('dataset').value,symbol:document.getElementById('symbol').value.trim().toUpperCase(),start_date:document.getElementById('startDate').value,end_date:document.getElementById('endDate').value,starting_equity:Number(document.getElementById('startingEquity').value),mode:document.getElementById('mode').value,base_timeframe:document.getElementById('baseTimeframe').value}; const job=await api('/api/backtests/jobs',{method:'POST',body:JSON.stringify(request)}); currentJobId=job.id; document.getElementById('cancelButton').disabled=false; updateJob(job); pollTimer=setInterval(pollJob,1000); }
  catch(error){ document.getElementById('jobMeta').innerHTML=`<span class="error-text">${escapeHtml(error.message)}</span>`; button.disabled=false; }
}
async function pollJob(){ if(!currentJobId)return; try{ const job=await api(`/api/backtests/jobs/${currentJobId}?include_result=true`); updateJob(job); if(['COMPLETED','FAILED','CANCELLED'].includes(job.status)){clearInterval(pollTimer);pollTimer=null;document.getElementById('runButton').disabled=false;document.getElementById('cancelButton').disabled=true;if(job.status==='COMPLETED'){backtestResult=job.result;renderBacktest();}} }catch(error){document.getElementById('jobMeta').textContent=error.message;} }
async function cancelJob(){ if(currentJobId) await api(`/api/backtests/jobs/${currentJobId}/cancel`,{method:'POST'}); }
function updateJob(job){ document.getElementById('jobName').textContent=`任务 ${job.id} · ${job.request.mode==='legacy'?'Legacy':'生产一致性'}`; document.getElementById('progressFill').style.width=`${job.progress||0}%`; document.getElementById('jobMeta').textContent=`${job.stage} · ${job.progress||0}%${job.error?' · '+job.error:''}`; }
function metric(label,value,sub='',className=''){return `<article class="panel metric"><label>${label}</label><strong class="${className}">${value}</strong><small>${sub}</small></article>`;}
function renderBacktest(){ const summary=backtestResult.summary||{}; document.getElementById('backtestMetrics').innerHTML=[metric('期末净值',Number(summary.ending_equity||0).toFixed(2)+' U',''),metric('总收益率',fmtPct(summary.total_return),'',cls(summary.total_return)),metric('最大回撤',fmtPct(summary.max_drawdown),'','negative'),metric('胜率',fmtPct(summary.win_rate)),metric('盈亏因子',Number(summary.profit_factor||0).toFixed(2)),metric('平均 R',fmtR(summary.average_r||0),'',cls(summary.average_r)),metric('交易次数',summary.trade_count||0),metric('手续费',Number(summary.fees||0).toFixed(2)+' U')].join(''); const curve=backtestResult.equity_curve||[]; lineChart('equityChart',curve,'equity','#059669'); let peak=-Infinity; const dd=curve.map(point=>{peak=Math.max(peak,Number(point.equity));return {...point,drawdown:peak>0?(Number(point.equity)-peak)/peak:0};}); lineChart('drawdownChart',dd,'drawdown','#dc2626',true); lineChart('pnlChart',backtestResult.analysis?.cumulative_curve||[],'value','#059669',true); renderBacktestResult(); renderDiagnostics(backtestResult.analysis?.diagnostics||[]); }
function renderBacktestResult(){ const host=document.getElementById('backtestResult'); if(!backtestResult){host.innerHTML='<div class="empty-state">配置参数并运行回测后查看结果</div>';return;} const analysis=backtestResult.analysis; if(!analysis){host.innerHTML=`<div class="empty-state">Legacy模式已完成：${(backtestResult.trades||[]).length} 条平仓记录。切换生产一致性模式可查看生命周期与插针诊断。</div>`;return;} if(activeBacktestTab==='Setup分析'){host.innerHTML='<div style="padding:14px"><h3 class="panel-title">Setup收益贡献</h3><div id="setupResultBars"></div></div>';barList('setupResultBars',analysis.setup_performance||[]);return;} if(activeBacktestTab==='生产 vs Legacy'){host.innerHTML='<div class="empty-state">选择相同数据分别运行生产一致性与Legacy模式，即可在任务历史中对照结果。当前版本保留两套固定模式，不会把规则叠加。</div>';return;} const rows=analysis.lifecycles||[]; host.innerHTML=`<div class="table-wrap"><table><thead><tr><th>币种</th><th>方向</th><th>Setup</th><th>质量</th><th>入场</th><th>出场</th><th>计划风险</th><th>实际R</th><th>MAE</th><th>MFE</th><th>持仓</th><th>出场原因</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${escapeHtml(row.symbol)}</td><td class="${row.side==='LONG'?'positive':'negative'}">${row.side}</td><td>${escapeHtml(row.setup_type)}</td><td>${escapeHtml(row.entry_quality)}</td><td>${Number(row.entry_price).toFixed(5)}</td><td>${Number(row.exit_price).toFixed(5)}</td><td>${Number(row.planned_risk_usdt).toFixed(2)} U</td><td class="${cls(row.realized_r)}">${fmtR(row.realized_r)}</td><td class="negative">-${Number(row.mae_r||0).toFixed(2)}R</td><td class="positive">${Number(row.mfe_r||0).toFixed(2)}R</td><td>${fmtDuration(row.holding_seconds)}</td><td title="${escapeHtml(row.exit_reason)}">${escapeHtml(row.exit_category)}</td></tr>`).join('')}</tbody></table></div>`; }
function renderDiagnostics(items){const host=document.getElementById('backtestDiagnostics');host.innerHTML=`<h3>诊断摘要</h3>${items.length?items.map(item=>`<div class="diagnostic"><strong class="${item.level==='positive'?'positive':item.level==='high'?'negative':''}">${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div>`).join(''):'<div class="empty-state">暂无明显异常</div>'}`;}
initDates();loadDatasets();toggleSource();renderBacktestTabs();renderBacktestResult();document.getElementById('backtestMetrics').innerHTML=['期末净值','总收益率','最大回撤','胜率','盈亏因子','平均 R','交易次数','手续费'].map(label=>metric(label,'--')).join('');lineChart('equityChart',[],'equity');lineChart('drawdownChart',[],'drawdown');lineChart('pnlChart',[],'value');
"""


REVIEW_BODY = r"""
<main class="page">
  <section class="panel toolbar"><div class="toolbar-grid review-toolbar-grid"><div class="field"><label>分析周期</label><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px"><input id="reviewStart" type="date"><input id="reviewEnd" type="date"></div></div><div class="field"><label>币种</label><select id="reviewSymbol" onchange="applyReviewFilters()"><option value="">全部</option></select></div><div class="field"><label>方向</label><select id="reviewSide" onchange="applyReviewFilters()"><option value="">全部</option><option>LONG</option><option>SHORT</option></select></div><div class="field"><label>Setup类型</label><select id="reviewSetup" onchange="applyReviewFilters()"><option value="">全部</option></select></div><div class="field"><label>信号质量</label><select id="reviewQuality" onchange="applyReviewFilters()"><option value="">全部</option><option>S</option><option>A</option><option>B</option><option>-</option></select></div><div class="field"><label>出场原因</label><select id="reviewExit" onchange="applyReviewFilters()"><option value="">全部</option><option>止盈</option><option>止损</option><option>结构退出</option><option>轮动</option><option>其他</option></select></div><div class="field run-field"><button class="primary" onclick="loadReview()">刷新复盘</button></div></div><div class="safety-note"><span>只读分析当前模拟账户成交，不会修改持仓、成交记录或风险状态。</span><span id="reviewUpdated">等待刷新</span></div></section>
  <section class="metrics" id="reviewMetrics"></section>
  <section class="charts"><article class="panel chart-panel"><h2 class="panel-title">累计净收益（按生命周期）</h2><div id="reviewCurve"></div></article><article class="panel chart-panel"><h2 class="panel-title">Setup贡献（按平均R）</h2><div id="reviewSetupBars"></div></article><article class="panel chart-panel"><h2 class="panel-title">出场原因分布</h2><div id="reviewExitBars"></div></article></section>
  <section class="tab-row" id="reviewTabs"></section><section class="result-layout"><div class="panel result-main" id="reviewResult"></div><aside class="panel diagnostics" id="reviewDiagnostics"><h3>复盘诊断</h3><div class="empty-state">正在读取成交记录</div></aside></section>
</main>
"""


REVIEW_SCRIPT = r"""
let reviewData=null, filteredLifecycles=[], activeReviewTab='交易生命周期', expandedLifecycle=null; const reviewTabs=['交易生命周期','Setup表现','止盈止损','加仓与轮动','异常交易'];
function initReviewDates(){const end=new Date();const start=new Date(end);start.setDate(start.getDate()-30);document.getElementById('reviewEnd').value=end.toISOString().slice(0,10);document.getElementById('reviewStart').value=start.toISOString().slice(0,10);}
function reviewMetric(label,value,className=''){return `<article class="panel metric"><label>${label}</label><strong class="${className}">${value}</strong></article>`;}
function renderReviewTabs(){document.getElementById('reviewTabs').innerHTML=reviewTabs.map(tab=>`<button class="${tab===activeReviewTab?'active':''}" onclick="selectReviewTab('${tab}')">${tab}</button>`).join('');}
function selectReviewTab(tab){activeReviewTab=tab;renderReviewTabs();renderReviewTable();}
async function loadReview(){try{reviewData=await api('/api/review/summary');document.getElementById('reviewUpdated').textContent=`复盘数据截至 ${new Date().toLocaleString()}`;populateReviewFilters();applyReviewFilters();}catch(error){document.getElementById('reviewResult').innerHTML=`<div class="empty-state error-text">${escapeHtml(error.message)}</div>`;}}
function populateReviewFilters(){const rows=reviewData.lifecycles||[];const options=(id,values)=>{const select=document.getElementById(id);const current=select.value;select.innerHTML='<option value="">全部</option>'+[...new Set(values.filter(Boolean))].sort().map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');select.value=current;};options('reviewSymbol',rows.map(row=>row.symbol));options('reviewSetup',rows.map(row=>row.setup_type));}
function applyReviewFilters(){if(!reviewData)return;const symbol=document.getElementById('reviewSymbol').value,side=document.getElementById('reviewSide').value,setup=document.getElementById('reviewSetup').value,quality=document.getElementById('reviewQuality').value,exit=document.getElementById('reviewExit').value,start=document.getElementById('reviewStart').value,end=document.getElementById('reviewEnd').value;filteredLifecycles=(reviewData.lifecycles||[]).filter(row=>(!symbol||row.symbol===symbol)&&(!side||row.side===side)&&(!setup||row.setup_type===setup)&&(!quality||row.entry_quality===quality)&&(!exit||row.exit_category===exit)&&(!start||row.closed_at.slice(0,10)>=start)&&(!end||row.closed_at.slice(0,10)<=end));renderReview();}
function summarize(rows){const pnls=rows.map(row=>Number(row.pnl||0)),rs=rows.map(row=>Number(row.realized_r||0)),wins=pnls.filter(v=>v>0),loss=Math.abs(pnls.filter(v=>v<0).reduce((a,b)=>a+b,0)),profit=wins.reduce((a,b)=>a+b,0);return{completed:rows.length,win_rate:rows.length?wins.length/rows.length:0,profit_factor:loss?profit/loss:profit,average_r:rows.length?rs.reduce((a,b)=>a+b,0)/rows.length:0,fees:reviewData.metrics?.fees||0,holding:rows.length?rows.reduce((a,b)=>a+Number(b.holding_seconds||0),0)/rows.length:0,mae:rows.length?rows.reduce((a,b)=>a+Number(b.mae_r||0),0)/rows.length:0,mfe:rows.length?rows.reduce((a,b)=>a+Number(b.mfe_r||0),0)/rows.length:0};}
function renderReview(){const metrics=summarize(filteredLifecycles);document.getElementById('reviewMetrics').innerHTML=[reviewMetric('完成交易（生命周期）',metrics.completed),reviewMetric('胜率',fmtPct(metrics.win_rate),'positive'),reviewMetric('盈亏因子',metrics.profit_factor.toFixed(2),'positive'),reviewMetric('平均 R',fmtR(metrics.average_r),cls(metrics.average_r)),reviewMetric('总手续费',Number(metrics.fees).toFixed(2)+' U','negative'),reviewMetric('平均持仓时间',fmtDuration(metrics.holding)),reviewMetric('平均 MAE','-'+metrics.mae.toFixed(2)+'R','negative'),reviewMetric('平均 MFE',metrics.mfe.toFixed(2)+'R','positive')].join('');let total=0;const curve=[...filteredLifecycles].sort((a,b)=>a.closed_at.localeCompare(b.closed_at)).map(row=>({timestamp:row.closed_at,value:(total+=Number(row.pnl||0))}));lineChart('reviewCurve',curve,'value','#059669',true);const setupGroups=groupRows(filteredLifecycles,'setup_type');barList('reviewSetupBars',setupGroups,'average_r');const exits=groupCounts(filteredLifecycles,'exit_category');barList('reviewExitBars',exits,'count');renderReviewTable();renderReviewDiagnostics();}
function groupRows(rows,key){const groups={};rows.forEach(row=>(groups[row[key]||'未分类']??=[]).push(row));return Object.entries(groups).map(([name,items])=>({name,count:items.length,pnl:items.reduce((a,b)=>a+Number(b.pnl||0),0),average_r:items.reduce((a,b)=>a+Number(b.realized_r||0),0)/items.length})).sort((a,b)=>b.average_r-a.average_r);}
function groupCounts(rows,key){const groups={};rows.forEach(row=>groups[row[key]||'其他']=(groups[row[key]||'其他']||0)+1);return Object.entries(groups).map(([name,count])=>({name,count})).sort((a,b)=>b.count-a.count);}
function toggleLifecycle(id){expandedLifecycle=expandedLifecycle===id?null:id;renderReviewTable();}
function renderReviewTable(){const host=document.getElementById('reviewResult');if(!reviewData){host.innerHTML='<div class="empty-state">正在读取成交记录</div>';return;}let rows=filteredLifecycles;if(activeReviewTab==='止盈止损')rows=rows.filter(row=>['止盈','止损'].includes(row.exit_category));if(activeReviewTab==='加仓与轮动')rows=rows.filter(row=>row.adds>0||row.exit_category==='轮动');if(activeReviewTab==='异常交易')rows=rows.filter(row=>row.realized_r<0&&row.mfe_r>=1);if(activeReviewTab==='Setup表现'){const groups=groupRows(rows,'setup_type');host.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Setup</th><th>样本</th><th>累计盈亏</th><th>平均R</th></tr></thead><tbody>${groups.map(item=>`<tr><td>${escapeHtml(item.name)}</td><td>${item.count}</td><td class="${cls(item.pnl)}">${fmtMoney(item.pnl)}</td><td class="${cls(item.average_r)}">${fmtR(item.average_r)}</td></tr>`).join('')}</tbody></table></div>`;return;}host.innerHTML=`<div class="table-wrap"><table><thead><tr><th>币种</th><th>方向</th><th>Setup</th><th>质量</th><th>入场时间</th><th>入场价</th><th>出场价</th><th>计划风险</th><th>实际R</th><th>MAE</th><th>MFE</th><th>持仓</th><th>出场原因</th></tr></thead><tbody>${rows.map(row=>`${lifecycleRow(row)}${expandedLifecycle===row.id?detailRow(row):''}`).join('')}</tbody></table></div>`;}
function lifecycleRow(row){return `<tr onclick="toggleLifecycle('${escapeHtml(row.id)}')" style="cursor:pointer"><td>${escapeHtml(row.symbol)}</td><td class="${row.side==='LONG'?'positive':'negative'}">${row.side}</td><td>${escapeHtml(row.setup_type)}</td><td>${escapeHtml(row.entry_quality)}</td><td>${new Date(row.opened_at).toLocaleString()}</td><td>${Number(row.entry_price).toFixed(5)}</td><td>${Number(row.exit_price).toFixed(5)}</td><td>${Number(row.planned_risk_usdt).toFixed(2)} U</td><td class="${cls(row.realized_r)}">${fmtR(row.realized_r)}</td><td class="negative">-${Number(row.mae_r).toFixed(2)}R</td><td class="positive">${Number(row.mfe_r).toFixed(2)}R</td><td>${fmtDuration(row.holding_seconds)}</td><td>${escapeHtml(row.exit_category)}</td></tr>`;}
function detailRow(row){return `<tr class="detail-row"><td colspan="13"><div class="trade-detail"><div><strong>入场记录</strong><p>${escapeHtml(row.entry_position||'无结构说明')}</p><p>初始止损 ${Number(row.stop_price).toFixed(5)}<br>TP1 ${Number(row.take_profit_1).toFixed(5)}<br>TP2 ${Number(row.take_profit_2).toFixed(5)}</p></div><div><strong>过程记录</strong><div class="timeline">${(row.timeline||[]).map(item=>`<div><span>${new Date(item.timestamp).toLocaleString()}</span><strong>${escapeHtml(item.action)}</strong><span>${Number(item.price).toFixed(5)} · ${escapeHtml(item.reason)}</span></div>`).join('')}</div></div><div><strong>结果</strong><p class="${cls(row.pnl)}">${fmtMoney(row.pnl)} / ${fmtR(row.realized_r)}</p><p>MAE -${Number(row.mae_r).toFixed(2)}R<br>MFE ${Number(row.mfe_r).toFixed(2)}R<br>${escapeHtml(row.exit_reason)}</p></div></div></td></tr>`;}
function renderReviewDiagnostics(){const host=document.getElementById('reviewDiagnostics'),items=reviewData?.diagnostics||[];host.innerHTML=`<h3>复盘诊断</h3>${items.map(item=>`<div class="diagnostic"><strong class="${item.level==='positive'?'positive':item.level==='high'?'negative':''}">${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div>`).join('')||'<div class="empty-state">当前筛选范围暂无明显异常</div>'}`;}
initReviewDates();renderReviewTabs();document.getElementById('reviewMetrics').innerHTML=['完成交易','胜率','盈亏因子','平均 R','总手续费','平均持仓','平均 MAE','平均 MFE'].map(label=>reviewMetric(label,'--')).join('');loadReview();
"""
