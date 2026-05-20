export const SECTOR_COLORS = {
    GEOPOLITICS: { hex: '#f87171', rgb: '248,113,113' },
    ECONOMY:     { hex: '#D4AF37', rgb: '212,175,55'  },
    FINANCE:     { hex: '#4ade80', rgb: '74,222,128'  },
    TECHNOLOGY:  { hex: '#7EB8E8', rgb: '126,184,232' },
    GENERAL:     { hex: '#C8C8D4', rgb: '200,200,212' },
    CULTURE:     { hex: '#C084FC', rgb: '192,132,252' },
};

const RSS_FEEDS = [
    { name: "WSJ Economy", url: "https://news.google.com/rss/search?q=WSJ+Economy&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Bloomberg Markets", url: "https://news.google.com/rss/search?q=Bloomberg+Markets&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Reuters Business", url: "https://news.google.com/rss/search?q=Reuters+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "FT Markets", url: "https://news.google.com/rss/search?q=Financial+Times+Markets&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "CNBC Business", url: "https://news.google.com/rss/search?q=CNBC+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "MarketWatch", url: "https://news.google.com/rss/search?q=MarketWatch&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Barron's", url: "https://news.google.com/rss/search?q=Barron's&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "The Economist", url: "https://news.google.com/rss/search?q=The+Economist&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Yahoo Finance", url: "https://news.google.com/rss/search?q=Yahoo+Finance&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Forbes Economy", url: "https://news.google.com/rss/search?q=Forbes+Economy&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Fed Press Releases", url: "https://news.google.com/rss/search?q=Federal+Reserve+Press+Releases&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Fed Speeches", url: "https://news.google.com/rss/search?q=Federal+Reserve+Speeches&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "FOMC News", url: "https://news.google.com/rss/search?q=FOMC&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Treasury Dept", url: "https://news.google.com/rss/search?q=US+Treasury+Department&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "SEC News", url: "https://news.google.com/rss/search?q=SEC+News&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Geopolitics News", url: "https://news.google.com/rss/search?q=Geopolitics&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "Global Trade", url: "https://news.google.com/rss/search?q=Global+Trade&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "Oil & Energy", url: "https://news.google.com/rss/search?q=Oil+Prices+Energy&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "China Economy", url: "https://news.google.com/rss/search?q=China+Economy&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "EU Economy", url: "https://news.google.com/rss/search?q=EU+Economy&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "Tech Crunch", url: "https://news.google.com/rss/search?q=TechCrunch&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "AI News", url: "https://news.google.com/rss/search?q=Artificial+Intelligence&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "Semiconductors", url: "https://news.google.com/rss/search?q=Semiconductors&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "The Verge", url: "https://news.google.com/rss/search?q=The+Verge&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "Wired", url: "https://news.google.com/rss/search?q=Wired&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "Fortune", url: "https://news.google.com/rss/search?q=Fortune+Magazine&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Business Insider", url: "https://news.google.com/rss/search?q=Business+Insider&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Quartz", url: "https://news.google.com/rss/search?q=Quartz&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Market Insider", url: "https://news.google.com/rss/search?q=Market+Insider&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "ZeroHedge", url: "https://news.google.com/rss/search?q=ZeroHedge&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Investopedia", url: "https://news.google.com/rss/search?q=Investopedia&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Seeking Alpha", url: "https://news.google.com/rss/search?q=Seeking+Alpha&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Morningstar", url: "https://news.google.com/rss/search?q=Morningstar&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Kiplinger", url: "https://news.google.com/rss/search?q=Kiplinger&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "RealClearMarkets", url: "https://news.google.com/rss/search?q=RealClearMarkets&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Economic Times", url: "https://news.google.com/rss/search?q=Economic+Times&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Wall Street Journal", url: "https://news.google.com/rss/search?q=Wall+Street+Journal&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "NYT Business", url: "https://news.google.com/rss/search?q=NYT+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "WaPo Business", url: "https://news.google.com/rss/search?q=Washington+Post+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Guardian Business", url: "https://news.google.com/rss/search?q=Guardian+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "BBC Business", url: "https://news.google.com/rss/search?q=BBC+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Al Jazeera Biz", url: "https://news.google.com/rss/search?q=Al+Jazeera+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Reuters Tech", url: "https://news.google.com/rss/search?q=Reuters+Technology&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "Bloomberg Tech", url: "https://news.google.com/rss/search?q=Bloomberg+Technology&hl=en-US&gl=US&ceid=US:en", sector: "TECHNOLOGY" },
    { name: "Nikkei Asia", url: "https://news.google.com/rss/search?q=Nikkei+Asia&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "SCMP", url: "https://news.google.com/rss/search?q=SCMP&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "Politico", url: "https://news.google.com/rss/search?q=Politico+Economy&hl=en-US&gl=US&ceid=US:en", sector: "GEOPOLITICS" },
    { name: "The Hill Biz", url: "https://news.google.com/rss/search?q=The+Hill+Business&hl=en-US&gl=US&ceid=US:en", sector: "ECONOMY" },
    { name: "Axios Markets", url: "https://news.google.com/rss/search?q=Axios+Markets&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Goldman Sachs", url: "https://news.google.com/rss/search?q=Goldman+Sachs&hl=en-US&gl=US&ceid=US:en", sector: "FINANCE" },
    { name: "Culture & Trends", url: "https://news.google.com/rss/search?q=Culture+Trends&hl=en-US&gl=US&ceid=US:en", sector: "CULTURE" },
    { name: "General News", url: "https://news.google.com/rss/search?q=World+News&hl=en-US&gl=US&ceid=US:en", sector: "GENERAL" }
];

const POSITIVE_WORDS = new Set(['growth','peace','recovery','cooperation','agreement','rise','gain','strong','breakthrough','approved','victory','record','expand','invest','alliance','stability','reform','progress','open','launch','success','profit','rally','surge','boost']);
const NEGATIVE_WORDS = new Set(['war','conflict','crisis','collapse','attack','bomb','kill','death','crash','deficit','sanction','threat','fail','loss','decline','protest','strike','cut','ban','terror','disease','flood','fire','drought','recession','inflation','violation']);
const VOLATILITY_WORDS = new Set(['breaking','urgent','sudden','shock','unprecedented','disruption','emergency','alert','surprise','unexpected','rapid','surge','spike','plunge','explosive','dramatic','critical','historic']);
const URGENCY_WORDS = new Set(['now','today','hours','minutes','immediately','latest','just','breaking']);

export const oracleState = {
    articles: [],
    sectorData: {},
    signalLog: [],
    lastFetch: 0,
    neuronPulses: [],
};

function scoreArticle(title, desc) {
    const text = (title + ' ' + desc).toLowerCase();
    const words = text.split(/\W+/);
    let pos = 0, neg = 0, vol = 0, urg = 0;
    
    words.forEach(w => {
        if (POSITIVE_WORDS.has(w))  pos++;
        if (NEGATIVE_WORDS.has(w))  neg++;
        if (VOLATILITY_WORDS.has(w)) vol++;
        if (URGENCY_WORDS.has(w))   urg++;
    });
    
    if (text.includes('this morning')) urg++;
    if (text.includes('this evening')) urg++;
    if (text.includes('live')) urg++;

    const total = pos + neg + 1;
    const sentiment = (pos - neg) / total;
    const volatility = Math.min(vol / 3, 1);
    const urgency    = Math.min(urg / 2, 1);
    const impact     = 0.6 * Math.min((vol + urg) / 5, 1) + 0.4 * Math.abs(sentiment);
    
    return { sentiment, volatility, urgency, impact };
}

async function fetchFeed(feed) {
    try {
        const res = await fetch(`https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(feed.url)}&count=10`, { cache: 'no-store' });
        if (res.ok) {
            const json = await res.json();
            if (json.status === 'ok' && json.items?.length) {
                return json.items.slice(0,8).map(item => {
                    const title   = (item.title||'').replace(/<[^>]+>/g,'').trim().slice(0,120);
                    const rawDesc = (item.description||item.content||'').replace(/<[^>]+>/g,'').trim().slice(0,200);
                    return { title, desc:rawDesc, url:item.link||'', source:feed.name, sector:feed.sector,
                             pubDate:item.pubDate||'', ...scoreArticle(title,rawDesc), fetchedAt:Date.now() };
                }).filter(a => a.title.length > 5);
            }
        }
    } catch(e) {}
    try {
        const res = await fetch(`https://feed2json.org/convert?url=${encodeURIComponent(feed.url)}`, { cache: 'no-store' });
        if (res.ok) {
            const json = await res.json();
            const items = json.items || json.entries || [];
            if (items.length) return items.slice(0,8).map(item => {
                const title   = (item.title?.value||item.title||'').replace(/<[^>]+>/g,'').trim().slice(0,120);
                const rawDesc = (item.summary?.value||item.content?.value||item.description||'').replace(/<[^>]+>/g,'').trim().slice(0,200);
                return { title, desc:rawDesc, url:item.url||item.link||'', source:feed.name, sector:feed.sector,
                         pubDate:item.date_published||'', ...scoreArticle(title,rawDesc), fetchedAt:Date.now() };
            }).filter(a => a.title.length > 5);
        }
    } catch(e) {}
    try {
        const res = await fetch(`https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(feed.url)}`, { cache: 'no-store' });
        if (res.ok) {
            const xmlText = await res.text();
            if (xmlText?.length > 200) { const items = parseRSSXML(xmlText, feed); if (items.length) return items; }
        }
    } catch(e) {}
    
    console.warn(`[ORACLE] All paths failed for: ${feed.name}`);
    return [];
}

function parseRSSXML(xml, feed) {
    try {
        const parser = new DOMParser();
        const doc = parser.parseFromString(xml, 'text/xml');
        const items = [...doc.querySelectorAll('item'), ...doc.querySelectorAll('entry')];
        return items.slice(0, 8).map(item => {
            const getText = (...tags) => {
                for (const tag of tags) {
                    const el = item.querySelector(tag);
                    if (el) return (el.textContent || el.getAttribute('url') || '').trim();
                }
                return '';
            };
            const title = getText('title').slice(0, 120);
            const rawDesc = getText('description', 'summary', 'content');
            const desc = rawDesc.replace(/<[^>]+>/g, '').trim().slice(0, 200);
            const link = getText('link') || item.querySelector('link')?.getAttribute('href') || '';
            const pubDate = getText('pubDate', 'published', 'updated');
            const scores = scoreArticle(title, desc);
            return { title, desc, url: link, source: feed.name, sector: feed.sector, pubDate, ...scores, fetchedAt: Date.now() };
        }).filter(a => a.title.length > 5);
    } catch(e) {
        return [];
    }
}

function computeSectorData(articles) {
    const data = {};
    Object.keys(SECTOR_COLORS).forEach(s => { data[s] = { sentiment: 0, count: 0, volatility: 0, articles: [] }; });
    articles.forEach(a => {
        const s = a.sector;
        if (!data[s]) return;
        data[s].count++;
        data[s].sentiment += a.sentiment;
        data[s].volatility += a.volatility;
        data[s].articles.push(a);
    });
    Object.keys(data).forEach(s => {
        if (data[s].count > 0) {
            data[s].sentiment /= data[s].count;
            data[s].volatility /= data[s].count;
        }
    });
    return data;
}

export function addOracleLog(msg, type = 'normal') {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    oracleState.signalLog.unshift({ time, msg, type });
    if (oracleState.signalLog.length > 20) oracleState.signalLog.pop();
    if (window._renderOracleLog) window._renderOracleLog();
}

export function formatAgo(ts) {
    const d = Date.now() - ts;
    if (d < 60000) return 'just now';
    if (d < 3600000) return Math.floor(d/60000) + 'm ago';
    return Math.floor(d/3600000) + 'h ago';
}

export async function fetchOracleData() {
    addOracleLog('⊕ Scanning all feeds + Oracle server in parallel…', 'watch');
    
    // Always use the secure Tailscale domain (it resolves locally on your machine too)
    const srvUrl = 'https://axium.tail02563d.ts.net/articles?limit=200';

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    const [srvResult, ...rssResults] = await Promise.allSettled([
        fetch(srvUrl, {
            signal: controller.signal
        }).then(r => {
            clearTimeout(timeoutId);
            return r.ok ? r.json() : Promise.reject(r.status);
        }).catch(e => {
            clearTimeout(timeoutId);
            return Promise.reject(e);
        }),
        ...RSS_FEEDS.map(f => fetchFeed(f))
    ]);

    const serverSentimentMap = new Map();
    let serverCount = 0;
    if (srvResult.status === 'fulfilled') {
        const serverArticles = srvResult.value.articles || [];
        serverCount = serverArticles.length;
        serverArticles.forEach(a => {
            const scores = scoreArticle(a.title, '');
            scores.sentiment = typeof a.sentiment === 'number' ? a.sentiment : scores.sentiment;
            const article = {
                title: a.title, desc: a.description || '', url: a.url || '',
                source: a.source || 'Oracle', sector: a.sector || 'GENERAL',
                pubDate: a.published || '', fetchedAt: a.published ? new Date(a.published).getTime() : Date.now(),
                _fromServer: true, ...scores
            };
            serverSentimentMap.set(a.title.slice(0, 40).toLowerCase(), article);
        });
        if (serverCount) addOracleLog('\u2726 Oracle server: ' + serverCount + ' FinBERT-scored signals', 'watch');
    } else {
        addOracleLog('\u26a0 Server offline — using RSS keyword scoring only', 'alert');
    }

    const rssArticles = rssResults.flatMap(r => r.status === 'fulfilled' ? r.value : []);
    const seen = new Set();
    const merged = [];

    serverSentimentMap.forEach(article => {
        const key = article.title.slice(0, 40).toLowerCase();
        if (!seen.has(key)) { seen.add(key); merged.push(article); }
    });

    rssArticles.forEach(a => {
        const key = a.title.slice(0, 40).toLowerCase();
        if (!seen.has(key)) {
            seen.add(key);
            merged.push(a);
        }
    });

    if (!merged.length) {
        addOracleLog('\u26a0 No signals retrieved from any source.', 'alert');
        return;
    }

    oracleState.articles = merged;
    oracleState.sectorData = computeSectorData(merged);
    oracleState.lastFetch = Date.now();

    const rssCount = merged.length - serverSentimentMap.size;
    addOracleLog('\u2234 ' + merged.length + ' total signals — ' + serverCount + ' server (FinBERT) + ' + rssCount + ' RSS');
    const top = [...merged].sort((a, b) => b.impact - a.impact)[0];
    if (top) addOracleLog('\u25b8 Top signal: [' + top.sector + '] ' + top.title.slice(0, 60) + '\u2026', top.sentiment < -0.1 ? 'alert' : 'watch');

    if (window._onOracleDataFetched) window._onOracleDataFetched(oracleState);
}