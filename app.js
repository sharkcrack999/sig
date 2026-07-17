const RAW="https://raw.githubusercontent.com/sharkcrack999/sig/main/signals.json";
const LOGRAW="https://raw.githubusercontent.com/sharkcrack999/sig/main/log.csv";
let LOG=[],PNL={};
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
function cdText(tf){
  const ms=TFMS[tf],now=Date.now();
  let next;
  if(tf==="1w"){
    const d=new Date(now),dow=d.getUTCDay();
    const days=(8-dow)%7||7;
    next=Date.UTC(d.getUTCFullYear(),d.getUTCMonth(),d.getUTCDate()+days);
  }else{next=Math.ceil(now/ms)*ms;}
  const left=next-now;
  if(left>=86400000){return Math.floor(left/86400000)+"d "+Math.floor(left%86400000/3600000)+"h";}
  if(left>=3600000){return Math.floor(left/3600000)+"h "+Math.floor(left%3600000/60000)+"m";}
  const m=Math.floor(left/60000),s=Math.floor(left%60000/1000);
  return m+":"+String(s).padStart(2,"0");
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
function
