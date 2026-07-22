from __future__ import annotations


def backtest_page() -> str:
    return HISTORICAL_BACKTEST_HTML


HISTORICAL_BACKTEST_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>历史回测 - AI量化交易平台</title>
  <style>
    :root{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;--navy:#111827;--page:#f6f7f9;--panel:#fff;--border:#dfe3e8;--line:#e8ebef;--muted:#667085;--blue:#2563eb;--green:#059669;--red:#dc2626;--amber:#d97706}
    *{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--page);color:#171717}body{font-size:12px;overflow:hidden}
    .header{height:52px;padding:7px 18px;background:var(--navy);color:#fff;display:flex;align-items:center;justify-content:space-between}.brand h1{font-size:19px;margin:0}.brand p{font-size:11px;color:#cbd5e1;margin:3px 0 0}.status{font-size:11px;color:#94a3b8;white-space:nowrap}
    .nav{height:38px;background:#fff;border-bottom:1px solid var(--border);display:flex;padding:0 16px;gap:6px}.nav a{display:flex;align-items:center;padding:0 13px;text-decoration:none;color:#64748b;font-weight:700;border-bottom:2px solid transparent}.nav a.active{color:var(--blue);border-color:var(--blue)}
    .metrics{height:76px;padding:8px 12px;display:grid;grid-template-columns:repeat(8,minmax(88px,1fr)) minmax(145px,1.35fr);gap:8px;overflow-x:auto}.metric,.panel{background:#fff;border:1px solid var(--border);border-radius:7px;box-shadow:0 1px 2px rgba(15,23,42,.04)}.metric{min-width:0;padding:8px 9px}.metric label{display:block;color:var(--muted);font-size:10px;white-space:nowrap}.metric strong{display:block;margin-top:7px;font-size:14px;white-space:nowrap}.positive{color:var(--green)!important}.negative{color:var(--red)!important}
    .workspace{height:calc(100vh - 166px);padding:0 12px 10px;display:grid;grid-template-columns:460px minmax(0,1fr);gap:8px}.left{padding:8px 10px;display:flex;flex-direction:column;min-height:0;overflow:hidden}.left h2,.right h2{font-size:14px;margin:0 0 8px}.date-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.field label{display:block;color:var(--muted);font-size:10px;margin-bottom:4px}input,button{font:inherit}input{width:100%;height:31px;border:1px solid #d6dbe2;border-radius:5px;padding:0 8px}button{height:34px;border:0;border-radius:5px;font-weight:700;cursor:pointer}.actions{display:grid;grid-template-columns:1.4fr 1fr;gap:7px;margin-top:8px}.primary{background:var(--blue);color:#fff}.secondary{background:#e9edf2;color:#344054}button:disabled{opacity:.5;cursor:not-allowed}.error{color:var(--red)!important}[hidden]{display:none!important}
    .replay-state{margin-top:10px;padding:10px;border:1px solid var(--line);border-radius:6px;background:#f8fafc}.replay-state-head{display:flex;justify-content:space-between;gap:8px}.replay-state strong{font-size:12px}.replay-state span,.replay-meta{color:var(--muted)}.progress-track{height:6px;margin:9px 0 8px;border-radius:999px;background:#e5e7eb;overflow:hidden}.progress-value{height:100%;width:0;background:var(--blue);transition:width .2s ease}.replay-meta{display:flex;justify-content:space-between;gap:8px;font-size:10px}.replay-message{margin-top:7px;color:#475467;line-height:1.45;overflow-wrap:anywhere}
    .analysis{margin-top:10px;padding-top:8px;border-top:1px solid var(--line);min-height:90px;overflow-y:scroll;scrollbar-gutter:stable;flex:1 1 auto}.analysis h3,.curve h3{font-size:12px;margin:0 0 5px}.analysis h3{position:sticky;top:0;z-index:1;padding-bottom:4px;background:#fff}.analysis-lines{color:#475467;line-height:1.5}.analysis-line{display:block;padding:6px 4px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}.analysis-line:last-child{border-bottom:0}.analysis-line strong{color:#111827}.analysis-line.loss-line strong{color:var(--red)}
    .curve{margin-top:10px;padding-top:8px;border-top:1px solid var(--line);height:clamp(220px,31vh,340px);min-height:220px;flex:0 0 clamp(220px,31vh,340px);overflow:hidden;display:flex;flex-direction:column}.curve-head{display:flex;justify-content:space-between;color:var(--muted)}#equityChart{flex:1;min-height:0}.chart-empty{height:100%;display:grid;place-items:center;color:#98a2b3}.chart-svg{width:100%;height:100%;display:block}.axis{stroke:#e5e7eb;stroke-width:1}.day-tick{fill:#667085;font-size:8px;text-anchor:middle}
    .right{display:grid;grid-template-rows:minmax(180px,36%) minmax(0,1fr);gap:8px;min-height:0}.right>.panel{min-height:0;padding:9px;overflow:hidden;display:flex;flex-direction:column}.section-head{display:flex;align-items:center;justify-content:space-between}.section-head small{color:var(--muted)}.table-wrap{min-height:0;overflow:auto;flex:1}table{width:100%;border-collapse:collapse;font-size:10px}th{position:sticky;top:0;z-index:1;background:#f8fafc;color:#596579;text-align:left;padding:6px;white-space:nowrap}td{padding:6px;border-top:1px solid var(--line);white-space:nowrap;vertical-align:top}.reason{white-space:normal;min-width:260px;line-height:1.35}.empty td{text-align:center;color:#98a2b3;padding:24px}
    @media(max-width:1100px){body{overflow:auto}.workspace{height:auto;grid-template-columns:1fr}.left{height:720px;min-height:720px}.right{height:900px}}
    @media(max-width:620px){.status{display:none}.workspace{padding:0 7px 8px}.date-grid{grid-template-columns:1fr}.right{height:1050px}.reason{min-width:200px}}
  </style>
</head>
<body>
  <header class="header"><div class="brand"><h1>AI 量化交易平台</h1><p>历史行情只驱动本地模拟账户，不会产生真实订单。</p></div><div class="status" id="globalStatus">读取系统状态中...</div></header>
  <nav class="nav"><a href="/">实时交易</a><a class="active" href="/backtest">历史回测</a></nav>
  <section class="metrics" id="metrics"></section>
  <main class="workspace">
    <aside class="panel left">
      <h2>历史回测</h2>
      <div class="date-grid"><div class="field"><label for="startDate">开始回测时间</label><input id="startDate" type="date"></div><div class="field"><label for="endDate">结束回测时间</label><input id="endDate" type="date"></div></div>
      <div class="actions"><button class="primary" id="runButton" onclick="runBacktest()">启动回测</button><button class="secondary" id="cancelButton" onclick="cancelBacktest()" disabled>取消回测</button></div>
      <section class="replay-state" id="replayState"><div class="replay-state-head"><strong id="replayStage">准备开始</strong><span id="replayProgress">0%</span></div><div class="progress-track"><div class="progress-value" id="progressValue"></div></div><div class="replay-meta"><span>当前模拟时间</span><span id="replayTime">--</span></div><div class="replay-message" id="replayMessage">选择时间范围后启动回测。</div></section>
      <section class="analysis" id="analysisSection" hidden><h3>分析总结</h3><div class="analysis-lines" id="failureSummary"><div class="analysis-line">暂无分析结果</div></div></section>
      <section class="curve" id="curveSection" hidden><div class="curve-head"><h3>总收益曲线</h3><span id="curvePeriod">开始时间 - 结束时间</span></div><div id="equityChart"><div class="chart-empty">暂无回测数据</div></div></section>
    </aside>
    <section class="right">
      <article class="panel"><div class="section-head"><h2>持仓</h2><small id="positionsAsOf">尚未回放</small></div><div class="table-wrap"><table><thead><tr><th>币种</th><th>方向</th><th>杠杆</th><th>入场</th><th>现价</th><th>数量</th><th>保证金</th><th>浮盈亏</th><th>收益率</th><th>止损</th><th>止盈</th><th>入场原因</th></tr></thead><tbody id="positions"><tr class="empty"><td colspan="12">启动回测后随进度显示持仓</td></tr></tbody></table></div></article>
      <article class="panel"><div class="section-head"><h2>已完成成交</h2><small id="tradesAsOf">尚未回放</small></div><div class="table-wrap"><table><thead><tr><th>币种</th><th>方向</th><th>杠杆</th><th>开仓均价</th><th>平仓均价</th><th>数量</th><th>止损</th><th>止盈</th><th>收益率</th><th>实现盈亏</th><th>手续费</th><th>开仓时间</th><th>平仓时间</th><th>入场位置</th><th>出场原因</th></tr></thead><tbody id="trades"><tr class="empty"><td colspan="15">启动回测后随进度显示已完成成交</td></tr></tbody></table></div></article>
    </section>
  </main>
  <script>
    const apiToken=localStorage.getItem('AI_TRADING_API_TOKEN');
    let jobId=null,pollTimer=null,result=null;

    async function api(path,options={}){
      const headers={'Content-Type':'application/json',...(options.headers||{})};
      if(apiToken)headers['X-API-Token']=apiToken;
      const response=await fetch(path,{...options,headers});
      if(!response.ok){
        let detail=`HTTP ${response.status}`;
        try{detail=(await response.json()).detail||detail}catch(_){}
        throw new Error(detail);
      }
      return response.json();
    }

    const esc=value=>String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    const pct=value=>`${(Number(value||0)*100).toFixed(2)}%`;
    const money=value=>`${Number(value||0)>=0?'+':''}${Number(value||0).toFixed(2)} U`;
    const cls=value=>Number(value||0)>=0?'positive':'negative';
    const amount=value=>Number(value||0).toFixed(2);
    const wholeUsdt=value=>String(Math.round(Number(value||0)));
    const displaySymbol=value=>{
      const raw=String(value||'').trim().toUpperCase().replace('/','').replace('-','');
      return raw.endsWith('USDT')?`${raw.slice(0,-4)}/USDT`:raw;
    };
    const sideText=value=>({LONG:'多',SHORT:'空'}[value]||value||'');
    const priceText=value=>{
      if(value===null||value===undefined||value==='')return'--';
      const n=Number(value);
      if(!Number.isFinite(n))return String(value);
      if(n===0)return'0';
      const abs=Math.abs(n);
      if(abs>=1){
        const digits=Math.floor(abs).toString().length;
        return n.toFixed(Math.max(0,4-digits));
      }
      return n.toFixed(Math.min(10,Math.max(0,Math.ceil(-Math.log10(abs))+3)));
    };
    const timeText=value=>{
      if(!value)return'--';
      const date=new Date(value);
      if(Number.isNaN(date.getTime()))return String(value);
      const parts=new Intl.DateTimeFormat('zh-CN',{
        timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',
        hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false
      }).formatToParts(date).reduce((acc,part)=>{acc[part.type]=part.value;return acc},{});
      return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`;
    };

    function initDates(){
      const end=new Date(),start=new Date(end);
      start.setDate(start.getDate()-14);
      document.getElementById('endDate').value=end.toISOString().slice(0,10);
      document.getElementById('startDate').value=start.toISOString().slice(0,10);
    }

    function metric(label,value='--',className=''){
      return `<article class="metric"><label>${label}</label><strong class="${className}">${value}</strong></article>`;
    }

    function renderEmptyMetrics(){
      document.getElementById('metrics').innerHTML=[
        '资金','可用','占用保证金','已实现','未实现','手续费','胜率','最大回撤','总收益'
      ].map(label=>metric(label)).join('');
    }

    function renderMetrics(summary={},account={}){
      if(account.equity===null||account.equity===undefined){
        renderEmptyMetrics();
        return;
      }
      const completed=Number(summary.trade_count||0);
      document.getElementById('metrics').innerHTML=[
        metric('资金',`${Number(account.equity||0).toFixed(2)} U`),
        metric('可用',`${Number(account.available_balance||0).toFixed(2)} U`),
        metric('占用保证金',`${Number(account.used_margin||0).toFixed(2)} U`),
        metric('已实现',money(account.realized_pnl),cls(account.realized_pnl)),
        metric('未实现',money(account.unrealized_pnl),cls(account.unrealized_pnl)),
        metric('手续费',money(-Number(account.fees_paid||0)),'negative'),
        metric('胜率',completed?pct(summary.win_rate):'--',completed?cls(summary.win_rate):''),
        metric('最大回撤',pct(summary.max_drawdown),Number(summary.max_drawdown||0)>0?'negative':''),
        metric('总收益',`${money(summary.total_pnl)} / ${pct(summary.total_return)}`,cls(summary.total_pnl))
      ].join('');
    }

    function setResultVisible(visible){
      document.getElementById('analysisSection').hidden=!visible;
      document.getElementById('curveSection').hidden=!visible;
    }

    function setReplayState(job){
      const progress=Math.max(0,Math.min(Number(job.progress||0),100));
      document.getElementById('replayProgress').textContent=`${progress}%`;
      document.getElementById('progressValue').style.width=`${progress}%`;
      document.getElementById('replayStage').textContent=job.stage||'处理中';
      document.getElementById('replayTime').textContent=job.virtual_time?timeText(job.virtual_time):'--';
      const message=document.getElementById('replayMessage');
      message.classList.remove('error');
      if(job.status==='COMPLETED')message.textContent='回测已完成，分析总结和总收益曲线已生成。';
      else if(job.status==='CANCELLED')message.textContent='回测已取消，当前显示停止前的持仓和成交快照。';
      else if(job.snapshot)message.textContent='顶部资金、持仓和已完成成交正在按历史回放进度更新。';
      else message.textContent='正在准备历史行情，进入回放后将显示持仓和已完成成交。';
    }

    function resetRunningView(){
      result=null;
      setResultVisible(false);
      renderEmptyMetrics();
      renderPositions([],'正在准备历史行情，暂未进入交易回放。');
      renderTrades([],'正在准备历史行情，暂未产生已完成成交。');
      document.getElementById('positionsAsOf').textContent='准备中';
      document.getElementById('tradesAsOf').textContent='准备中';
    }

    async function runBacktest(){
      clearInterval(pollTimer);
      resetRunningView();
      document.getElementById('runButton').disabled=true;
      document.getElementById('cancelButton').disabled=false;
      setReplayState({progress:0,stage:'提交回测任务',status:'RUNNING'});
      try{
        const payload={
          start_date:document.getElementById('startDate').value,
          end_date:document.getElementById('endDate').value
        };
        const job=await api('/api/backtests/jobs',{method:'POST',body:JSON.stringify(payload)});
        jobId=job.id;
        updateJob(job);
        pollTimer=setInterval(pollJob,700);
      }catch(error){
        showError(error.message);
        finishControls();
      }
    }

    async function pollJob(){
      if(!jobId)return;
      try{
        const job=await api(`/api/backtests/jobs/${jobId}?include_result=true`);
        updateJob(job);
        if(['COMPLETED','FAILED','CANCELLED'].includes(job.status)){
          clearInterval(pollTimer);
          pollTimer=null;
          finishControls();
          if(job.status==='COMPLETED'){
            result=job.result;
            renderResult();
          }else if(job.error){
            showError(job.error);
          }
        }
      }catch(error){
        showError(error.message);
        clearInterval(pollTimer);
        pollTimer=null;
        finishControls();
      }
    }

    async function cancelBacktest(){
      if(!jobId)return;
      try{
        const job=await api(`/api/backtests/jobs/${jobId}/cancel`,{method:'POST'});
        updateJob(job);
      }catch(error){
        showError(error.message);
      }
    }

    function finishControls(){
      const button=document.getElementById('runButton');
      button.disabled=false;
      button.textContent='启动回测';
      document.getElementById('cancelButton').disabled=true;
    }

    function updateJob(job){
      document.getElementById('runButton').textContent='回测运行中';
      setReplayState(job);
      if(!job.snapshot)return;
      const snapshot=job.snapshot;
      const account=snapshot.account||{};
      renderMetrics(snapshot.summary||{},account);
      const asOf=job.virtual_time?`截至 ${timeText(job.virtual_time)}`:'回放中';
      document.getElementById('positionsAsOf').textContent=asOf;
      document.getElementById('tradesAsOf').textContent=asOf;
      renderPositions(account.positions||[],'截至当前模拟时间无持仓');
      renderTrades(account.fills||[],'截至当前模拟时间尚未产生已完成成交');
    }

    function showError(message){
      setResultVisible(false);
      document.getElementById('replayStage').textContent='回测失败';
      const host=document.getElementById('replayMessage');
      host.textContent=String(message||'未知错误');
      host.classList.add('error');
    }

    function renderResult(){
      const summary=result.summary||{},account=result.account||{},analysis=result.analysis||{};
      renderMetrics(summary,account);
      setResultVisible(true);
      renderAnalysis(analysis);
      const period=result.period||{};
      document.getElementById('curvePeriod').textContent=`${String(period.start||'').slice(0,10)} - ${String(period.end||'').slice(0,10)}`;
      renderCurve(result.equity_curve||[],period);
      document.getElementById('positionsAsOf').textContent='回测结束';
      document.getElementById('tradesAsOf').textContent='回测结束';
      renderPositions(account.positions||[],'回测结束时无持仓');
      renderTrades(account.fills||[],'本区间没有已完成成交');
    }

    function renderAnalysis(analysis){
      const host=document.getElementById('failureSummary'),rows=analysis.symbol_summaries||[];
      host.classList.remove('error');
      if(!rows.length){
        host.innerHTML=`<div class="analysis-line">${esc(analysis.failure_summary||'本区间暂无完成交易')}</div>`;
        return;
      }
      host.innerHTML=rows.map(row=>`<div class="analysis-line ${Number(row.pnl||0)<0?'loss-line':''}"><strong>${esc(displaySymbol(row.symbol))}</strong>：${esc(row.text||'暂无分析')}</div>`).join('');
    }

    const trendText={
      CHOP:'震荡',TREND_LONG:'多头趋势',TREND_SHORT:'空头趋势',
      ONE_WAY_UP:'单边上涨',ONE_WAY_DOWN:'单边下跌'
    };
    const riskText={
      NORMAL:'风险正常',LONG_CROWD:'多头拥挤',SHORT_CROWD:'空头拥挤',
      OI_ABNORMAL:'持仓量异常',FUNDING_HOT:'资金费率过热'
    };

    function setupText(context,side){
      const code=String(context.setup_type||context.entry_setup||'').toUpperCase();
      if(code.includes('DISTRIBUTION'))return'高位派发结构';
      if(code.includes('OI_VALLEY'))return'4小时持仓量洼地反转';
      if(code.includes('SQUEEZE'))return'15分钟挤压回踩';
      if(code.includes('PULLBACK')||code.includes('RETEST'))return side==='LONG'?'趋势回踩做多':'趋势反抽做空';
      if(code.includes('STRUCTURE'))return side==='LONG'?'多头结构确认':'空头结构确认';
      if(code.includes('BREAKOUT'))return'突破回踩确认';
      if(code.includes('BREAKDOWN'))return'跌破反抽确认';
      return'';
    }

    function entryReasonText(row){
      const raw=String(row.entry_reason||row.reason||'');
      if(raw.toLowerCase().includes('manual')||raw.includes('手动'))return'手动开仓';
      const context=row.entry_context&&typeof row.entry_context==='object'?row.entry_context:{};
      const parts=['自动开仓'];
      if(row.entry_score!==null&&row.entry_score!==undefined&&String(row.entry_score)!==''){
        parts.push(`评分 ${row.entry_score}`);
      }
      const setup=setupText(context,row.side);
      if(setup)parts.push(setup);
      const trend=trendText[context.trend_state||context.regime];
      if(trend)parts.push(trend);
      const risk=riskText[context.risk_state];
      if(risk)parts.push(risk);
      return [...new Set(parts)].join('；');
    }

    function exitReasonText(value){
      const reason=String(value||'').trim();
      if(!reason)return'策略退出：触发既定离场条件';
      const lower=reason.toLowerCase();
      if(lower.includes('rotation exit'))return'调仓退出：标的相对优势下降';
      if(lower.includes('risk exit'))return'风险退出：触发持仓风险控制';
      if(lower.includes('manual'))return'手动平仓';
      if(lower.includes('funding settlement'))return'资金费结算';
      let prefix='策略退出';
      if(lower.startsWith('stop loss'))prefix='止损';
      else if(lower.startsWith('take profit'))prefix='止盈';
      else if(lower.includes('time exit'))prefix='时间退出';
      const details=[
        ['target 1 reached','第一止盈目标达成'],
        ['target 2 reached','第二止盈目标达成'],
        ['protected stop slipped below entry','保护止损低于开仓价成交'],
        ['protected stop after profit lock','锁定利润后触发保护止损'],
        ['signal direction or structure failed','信号方向或结构失效'],
        ['atr volatility hard stop','触发波动率硬止损'],
        ['15m entry structure stop','15分钟入场结构失效'],
        ['breakout protection stop','突破保护止损'],
        ['short trend support protection stop','空头趋势支撑保护止损'],
        ['strong trend ema50 structure invalidated','强趋势均线结构失效'],
        ['floating profit drawdown protection','浮盈回撤保护'],
        ['profit drawdown','盈利回撤保护'],
        ['near 4h resistance with profit protection','接近4小时压力位并保护利润'],
        ['near 4h support with profit protection','接近4小时支撑位并保护利润'],
        ['4h support plus short exhaustion confirmed','4小时支撑与空头衰竭确认'],
        ['4h resistance plus long exhaustion confirmed','4小时压力与多头衰竭确认'],
        ['body closed below support or ema/boll zone','实体收盘跌破支撑或均线区域'],
        ['body closed above resistance or ema/boll zone','实体收盘突破压力或均线区域'],
        ['structure invalidated','交易结构失效'],
        ['trend late stage','趋势进入末期'],
        ['efficiency','持仓效率下降']
      ];
      const matched=details.find(([needle])=>lower.includes(needle));
      if(matched)return`${prefix}：${matched[1]}`;
      if(prefix==='止损')return'止损：交易条件失效';
      if(prefix==='止盈')return'止盈：达到目标或保护条件';
      if(prefix==='时间退出')return'时间退出：持仓效率不足';
      if(/[\u4e00-\u9fff]/.test(reason)&&!/[a-z]{4,}/i.test(reason))return reason;
      return'策略退出：触发既定离场条件';
    }

    function renderPositions(rows,emptyText='当前无持仓'){
      document.getElementById('positions').innerHTML=rows.length?rows.map(row=>`<tr><td>${esc(displaySymbol(row.symbol))}</td><td class="${row.side==='LONG'?'positive':'negative'}">${esc(sideText(row.side))}</td><td>${row.leverage||0}x</td><td>${priceText(row.entry_price)}</td><td>${priceText(row.mark_price)}</td><td>${wholeUsdt(row.notional)}</td><td>${amount(row.margin_usdt)}</td><td class="${cls(row.unrealized_pnl)}">${amount(row.unrealized_pnl)}</td><td class="${cls(row.unrealized_pnl_pct_on_margin)}">${pct(row.unrealized_pnl_pct_on_margin)}</td><td>${priceText(row.stop_price)}</td><td>${priceText(row.take_profit_2)}</td><td class="reason">${esc(entryReasonText(row))}</td></tr>`).join(''):`<tr class="empty"><td colspan="12">${esc(emptyText)}</td></tr>`;
    }

    function renderTrades(fills,emptyText='暂无已完成成交'){
      const rows=[...(fills||[])].filter(row=>row.action==='CLOSE').reverse();
      document.getElementById('trades').innerHTML=rows.length?rows.map(row=>`<tr><td>${esc(displaySymbol(row.symbol))}</td><td class="${row.side==='LONG'?'positive':'negative'}">${esc(sideText(row.side))}</td><td>${row.leverage||0}x</td><td>${priceText(row.entry_price||row.price)}</td><td>${priceText(row.price)}</td><td>${wholeUsdt(Number(row.price||0)*Number(row.quantity||0))}</td><td>${priceText(row.stop_price)}</td><td>${priceText(row.take_profit_2)}</td><td class="${cls(row.return_pct)}">${pct(row.return_pct)}</td><td class="${cls(row.realized_pnl)}">${amount(row.realized_pnl)}</td><td>${amount(row.fee)}</td><td>${timeText(row.opened_at)}</td><td>${timeText(row.closed_at)}</td><td class="reason">${esc(String(row.entry_position||'--').replace(/\s*\r?\n\s*/g,' '))}</td><td class="reason">${esc(exitReasonText(row.reason))}</td></tr>`).join(''):`<tr class="empty"><td colspan="15">${esc(emptyText)}</td></tr>`;
    }

    function renderCurve(points,period={}){
      const host=document.getElementById('equityChart');
      if(points.length<2){
        host.innerHTML='<div class="chart-empty">暂无可绘制数据</div>';
        return;
      }
      const w=460,h=170,l=42,r=7,t=8,b=30,values=points.map(p=>Number(p.equity||0));
      let min=Math.min(...values),max=Math.max(...values);
      if(max===min){max+=1;min-=1}
      const x=i=>l+i*(w-l-r)/Math.max(points.length-1,1);
      const y=v=>t+(max-v)*(h-t-b)/(max-min);
      const line=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
      const grid=[0,.5,1].map(q=>{
        const v=max-(max-min)*q,yy=y(v);
        return `<line class="axis" x1="${l}" x2="${w-r}" y1="${yy}" y2="${yy}"/><text x="2" y="${yy+3}" font-size="9" fill="#667085">${v.toFixed(1)}</text>`;
      }).join('');
      const start=String(period.start||points[0].timestamp).slice(0,10);
      const end=String(period.end||points.at(-1).timestamp).slice(0,10);
      const parts=start.split('-').map(Number),endParts=end.split('-').map(Number);
      const startDay=Date.UTC(parts[0],parts[1]-1,parts[2]),endDay=Date.UTC(endParts[0],endParts[1]-1,endParts[2]);
      const dayCount=Math.max(Math.round((endDay-startDay)/86400000)+1,1);
      const ticks=Array.from({length:dayCount},(_,i)=>{
        const day=new Date(startDay+i*86400000),xx=l+i*(w-l-r)/Math.max(dayCount-1,1);
        return `<line class="axis" x1="${xx}" x2="${xx}" y1="${h-b}" y2="${h-b+4}"/><text class="day-tick" x="${xx}" y="${h-7}">${day.getUTCDate()}</text>`;
      }).join('');
      host.innerHTML=`<svg class="chart-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${grid}${ticks}<polyline points="${line}" fill="none" stroke="#059669" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
    }

    async function refreshHeader(){
      try{
        const h=await api('/api/health');
        document.getElementById('globalStatus').textContent=`行情与持仓管理${h.running?'运行中':'未运行'} | 新开仓${h.auto_trade?'允许':'禁止'} | ${new Date().toLocaleString()}`;
      }catch(_){}
    }

    initDates();
    renderEmptyMetrics();
    setResultVisible(false);
    refreshHeader();
    setInterval(refreshHeader,15000);
  </script>
</body>
</html>"""
