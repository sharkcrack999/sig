const RAW="https://raw.githubusercontent.com/sharkcrack999/sig/main/signals.json";
const LOGRAW="https://raw.githubusercontent.com/sharkcrack999/sig/main/log.csv";
const BINANCE="https://data-api.binance.vision/api/v3/ticker/price";
let LOG=[],PNL={},LIVE={};
const FEE=0.0015;
const TFS=["5m","30m","1h","1d","1w"];
const TFN={"5m":"5 MIN","30m":"30 MIN","1h":"1 HOUR","1d":"1 DAY","1w":"1 WEEK"};
const TFMS={"5m":300000,"30m":1800000,"1h":3600000,"1d":86400000,"1w":604800000};
const CO=["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","TRX","AVAX","LINK"];
let DATA=null,SEL=(location.hash||"").replace("#","").toUpperCase()||"ALL";
let TFSEL="ALL";
const short=s=>s.replace("USDT","");
const cls=c=>c.includes("BUY")?"buy":c.includes("SELL")?"sell":"neut";
const arrow=c=>c==="STRONG BUY"?"\u25B2\u25B2":c==="BUY"?"\u25B2":c==="STRONG SELL"?"\u25BC\u25BC":c==="SELL"?"\u25BC":"\u2014";
const fp=p=>"$"+(p>=1?p.toLocaleString(undefined,{maximumFractionDigits:2}):p.toFixed(5));
const fg=x=>(x>=0?"+":"")+(x*100).toFixed(2)+"%";
const ago=t=>{const m=Math.max(0,(Date.now()-t)/60000);
  return m<1?"now":m<60?Math.round(m)+"m":Math.round(m/60)+"h"};
const mmss=ms=>{if(ms<=0)return "0:00";
  if(ms>=86400000)return Math.floor(ms/86400000)+"d "+Math.floor(ms%86400000/3600000)+"h";
  if(ms>=3600000)return Math.floor(ms/3600000)+"h "+Math.floor(ms%3600000/60000)+"m";
  return Math.floor(ms/60000)+":"+String(Math.floor(ms%60000/1000)).padStart(2,"0")};
function cdText(tf){
  const ms=TFMS[tf],now=Date.now();
  let next;
  if(tf==="1w"){
    const d=new Date(now),dow=d.getUTCDay();
    const days=(8-dow)%7||7;
    next=Date.UTC(d.getUTCFullYear(),d.getUTCMonth(),d.getUTCDate()+days);
  }else{next=Math.ceil(now/ms)*ms;}
  return mmss(next-now);
}
function tickTimers(){
  for(const tf of TFS){
    const h=document.getElementById("cdh-"+tf);
    if(h)h.textContent="closes "+cdText(tf);
    const c=document.getElementById("cdc-"+tf);
    if(c)c.textContent=cdText(tf);
  }
}
function relFor(key){
  const lv=(DATA.latest||{})[key];
  if(lv&&lv.rel!=null&&lv.reln>=5)return{r:lv.rel,n:lv.reln};
  const x=(DATA.history||[]).filter(e=>e.key===key&&
    (e.result===true||e.result===false)).slice(-30);
  if(x.length<5)return null;
  return {r:x.filter(e=>e.result===true).length/x.length,n:x.length};
}
function relHTML(key){
  const v=relFor(key);
  if(!v)return '<div class="rel">rel \u2014</div>';
  const c=v.r>=0.55?"r-g":v.r>=0.45?"r-y":"r-r";
  return '<div class="rel '+c+'">rel '+(v.r*100).toFixed(0)+'% \u00B7 '+v.n+'</div>';
}
function livePx(sym){return LIVE[sym]||null;}
function pick(c){SEL=c;location.hash=c==="ALL"?"":c.toLowerCase();
  if(DATA)render(DATA);}
function pickTF(t){TFSEL=t;if(DATA)render(DATA);}
function render(d){
  DATA=d;
  const coins={};
  for(const [k,v] of Object.entries(d.latest||{})){
    const [sym,tf]=k.split(":");
    (coins[short(sym)]=coins[short(sym)]||{})[tf]={...v,key:k,sym:sym};
  }
  const names=[...CO.filter(c=>coins[c]),
               ...Object.keys(coins).filter(c=>!CO.includes(c))];
  document.getElementById("tabs").innerHTML=["ALL",...names].map(c=>
    '<button class="'+(c===SEL?"on":"")+'" onclick="pick(\''+c+'\')">'+c+'</button>').join("");
  if(!names.length)return;
  if(SEL==="ALL"){
    const l5=Object.entries(d.latest||{}).filter(([k])=>k.endsWith(":5m"));
    const bear=l5.filter(([,v])=>v.p<0.5).length;
    const macro=(bear>=l5.length*0.8||bear<=l5.length*0.2)?
      " \u00B7 board is one macro trade right now, not "+l5.length+" independent signals":"";
    const breadth=l5.length?'<div class="empty" style="padding:6px 2px 12px">Market breadth: <b class="'+
      (bear>l5.length/2?"c-sell":"c-buy")+'">'+bear+"/"+l5.length+
      ' coins bearish</b> on 5m'+macro+'</div>':"";
    let rows="";
    for(const n of names){
      const tfs=coins[n],any=Object.values(tfs)[0];
      const px=livePx(any.sym)||any.price;
      let cells="";
      for(const t of TFS){
        const v=tfs[t];
        if(!v){cells+='<td class="cell"><div class="a neut">\u00B7</div></td>';continue;}
        cells+='<td class="cell"><div class="a '+cls(v.call)+'">'+arrow(v.call)+
          '</div><div class="p mono">'+(v.p*100).toFixed(0)+'%</div>'+relHTML(v.key)+'</td>';
      }
      rows+='<tr onclick="pick(\''+n+'\')"><td class="sym"><b>'+n+
        '</b><span class="px mono">'+fp(px)+'</span></td>'+cells+'</tr>';
    }
    document.getElementById("board").innerHTML=breadth+
      '<table class="mx"><thead><tr><th>Market</th>'+
      TFS.map(t=>'<th>'+TFN[t]+'<div class="cd mono" id="cdh-'+t+'"></div></th>').join("")+
      '</tr></thead><tbody>'+rows+'</tbody></table>';
  }else{
    const tfs=coins[SEL];
    if(!tfs){document.getElementById("board").innerHTML=
      '<div class="empty">No data for '+SEL+' yet.</div>';}
    else{
      const any=Object.values(tfs)[0];
      const px=livePx(any.sym)||any.price;
      let cards="";
      for(const t of TFS){
        if(!tfs[t])continue;
        const v=tfs[t],rel=relFor(v.key);
        const relLine=rel?('Reliability <b class="'+(rel.r>=0.55?"c-buy":rel.r>=0.45?"":"c-sell")+
          '">'+(rel.r*100).toFixed(0)+'%</b> of '+rel.n+' calls'):'Reliability <b>collecting\u2026</b>';
        const lp=livePx(v.sym);
        const liveLine=(lp&&v.call!=="NEUTRAL")?
          ('Live: <b class="'+((v.call.includes("BUY")?1:-1)*(lp/v.price-1)>=0?"g-pos":"g-neg")+'">'+
           fg((v.call.includes("BUY")?1:-1)*(lp/v.price-1))+'</b> so far'):"";
        cards+='<div class="card"><div class="tf"><span>'+TFN[t]+
          '</span><span class="mono">closes <i style="font-style:normal" id="cdc-'+t+'"></i></span></div>'+
          '<div class="gauge"><i style="left:'+(v.p*100).toFixed(1)+'%"></i></div>'+
          '<div class="call '+cls(v.call)+'">'+v.call+'</div>'+
          '<div class="meta mono">P(up) <b>'+(v.p*100).toFixed(1)+'%</b> \u00B7 '+fp(v.price)+'</div>'+
          '<div class="meta mono">'+relLine+'</div>'+
          (liveLine?'<div class="meta mono">'+liveLine+'</div>':"")+'</div>';
      }
      document.getElementById("board").innerHTML=
        '<div class="coinhead"><b>'+SEL+'</b><span class="mono" style="color:var(--faint);font-size:12px">/ USDT</span>'+
        '<span class="back" onclick="pick(\'ALL\')">\u2190 ALL</span>'+
        '<span class="px mono">'+fp(px)+'</span></div><div class="cards">'+cards+'</div>';
    }
  }
  const sc=LOG.filter(r=>(r.result==="hit"||r.result==="miss")&&
    (SEL==="ALL"||r.key.startsWith(SEL)));
  const hits=sc.filter(r=>r.result==="hit").length;
  const st=sc.filter(r=>r.strong==="1");
  const sth=st.filter(r=>r.result==="hit").length;
  let net=0;
  for(const r of sc){
    const g=(+r.dir)*((+r.next_price)/(+r.price)-1);
    if(isFinite(g))net+=g;
  }
  const pend=(d.history||[]).filter(e=>e.result===null&&e.dir!==0&&
    (SEL==="ALL"||short(e.key.split(":")[0])===SEL)).length;
  const pc=(h,n)=>n?((100*h/n).toFixed(1)+"%"):"\u2014";
  const netAfter=net-FEE*sc.length;
  document.getElementById("statbar").innerHTML=
    '<div class="stat"><div class="k">Signals scored</div><div class="v mono">'+sc.length+
    ' <small>'+pend+' pending</small></div></div>'+
    '<div class="stat"><div class="k">Accuracy</div><div class="v mono">'+pc(hits,sc.length)+'</div></div>'+
    '<div class="stat"><div class="k">Strong accuracy</div><div class="v mono">'+pc(sth,st.length)+'</div></div>'+
    '<div class="stat"><div class="k">Gain after fees</div><div class="v mono '+
    (netAfter>=0?"g-pos":"g-neg")+'">'+(sc.length?fg(netAfter):"\u2014")+
    ' <small>'+(sc.length?"gross "+fg(net):"")+'</small></div></div>';
  tickTimers();
  PNL={};
  for(const r of LOG){
    const g=(+r.dir)*((+r.next_price)/(+r.price)-1);
    if(isFinite(g))PNL[r.key+"|"+r.t]=g;
  }
  document.getElementById("tfchips").innerHTML=["ALL",...TFS].map(t=>
    '<button class="'+(t===TFSEL?"on":"")+'" onclick="pickTF(\''+t+'\')">'+
    (t==="ALL"?"ALL":TFN[t])+'</button>').join("");
  const hist=(d.history||[]).filter(e=>
    (SEL==="ALL"||short(e.key.split(":")[0])===SEL)&&
    (TFSEL==="ALL"||e.key.endsWith(":"+TFSEL)))
    .slice(-15).reverse();
  let hrows="";
  for(const e of hist){
    const [sym,tf]=e.key.split(":");
    const rowCls=e.result===true?"rh":e.result===false?"rm":"";
    const lp=livePx(sym);
    let pill,gain;
    if(e.result===null){
      const locks=e.t+TFMS[tf]-Date.now();
      pill='<span class="pill pend">'+(locks>0?("locks "+mmss(locks)):"scoring\u2026")+'</span>';
      if(e.dir!==0&&lp){
        const g=e.dir*(lp/e.price-1);
        gain='<span class="mono '+(g>=0?"g-pos":"g-neg")+'" style="opacity:.75">'+fg(g)+' live</span>';
      }else{gain='<span class="mono" style="color:var(--faint)">\u00B7\u00B7\u00B7</span>';}
    }else if(e.result==="na"){pill="";gain="";}
    else{
      pill=e.result?'<span class="pill ok">\u2713</span>':'<span class="pill bad">\u2715</span>';
      const g=PNL[e.key+"|"+e.t];
      gain=g===undefined?"\u2014":'<span class="mono '+(g>=0?"g-pos":"g-neg")+'">'+fg(g)+'</span>';
    }
    hrows+='<tr class="'+rowCls+'" onclick="pick(\''+short(sym)+'\')"><td>'+short(sym)+
      ' <span class="mono" style="color:var(--faint)">'+tf+'</span></td>'+
      '<td class="mono">'+ago(e.t)+'</td>'+
      '<td class="c-'+cls(e.call)+'">'+arrow(e.call)+' '+e.call+'</td>'+
      '<td>'+gain+'</td><td>'+pill+'</td></tr>';
  }
  document.getElementById("hist").innerHTML=hrows||
    '<tr><td class="empty" colspan="5">No calls yet</td></tr>';
  const ts=Math.max(0,...Object.values(d.latest||{}).map(v=>v.t));
  document.getElementById("upd").textContent=ts?("engine "+ago(ts)+" ago \u00B7 prices live"):"no data";
}
async function pollPrices(){
  try{
    const arr=await fetch(BINANCE).then(r=>r.json());
    for(const x of arr){LIVE[x.symbol]=parseFloat(x.price);}
    if(DATA)render(DATA);
  }catch(e){}
}
async function load(){
  try{
    const d=await fetch(RAW+"?x="+Date.now()).then(r=>r.json());
    try{
      const t=await fetch(LOGRAW+"?x="+Date.now()).then(r=>r.text());
      const rows=t.trim().split("\n");const head=rows[0].split(",");
      LOG=rows.slice(1).map(l=>{const c=l.split(",");const o={};
        head.forEach((h,i)=>o[h]=c[i]);return o;});
    }catch(e){LOG=[];}
    render(d);
    document.getElementById("dot").style.background="var(--buy)";
  }catch(e){
    document.getElementById("dot").style.background="var(--sell)";
    document.getElementById("upd").textContent="waiting for data";
  }
}
load();setInterval(load,30000);
setInterval(tickTimers,500);
pollPrices();setInterval(pollPrices,15000);
