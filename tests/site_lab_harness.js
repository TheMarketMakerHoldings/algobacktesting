// Test helper for tests/test_site_lab.py: runs the page's stop lab on the inputs it is given.
// The test prepends the page's own code (statistics, importer and lab) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const page = input.page, out = {};
const parsed = parseBars(input.bars), example = readExport(page.trades).table;
out.bars = { n: parsed.bars.length, barMs: parsed.barMs, first: parsed.bars[0], last: parsed.bars[parsed.bars.length - 1] };
out.scans = input.scans.map(function (s) {
  const filter = { col: example.header.indexOf("Strategy"), value: s.strategy };
  const entries = tradeEntries(example, filter);
  const spec = INSTRUMENTS[instrumentRoot(entries[0].symbol)];
  const al = alignEntries(entries, parsed.bars, parsed.barMs, spec.tick);
  const scan = stopScan(al, parsed.bars, {
    spec: spec, stops: s.stops, targetTicks: s.target, breakevenTicks: s.breakeven, maxHoldMs: 240 * 60000,
    gapMs: 30 * 60000, setups: 5, feePerContract: 4.12
  });
  return { matched: al.matched.length, entries: entries.length, shift: al.shiftHours, atOpen: al.matched.filter(function (m) { return m.atOpen; }).length,
           trials: scan.trials, rows: scan.rows.map(function (r) { return { ticks: r.ticks, values: r.values, chance: r.chance }; }) };
});
out.cases = input.cases.map(function (c) {
  const times = c.bars.map(function (b) { return b[0]; });
  const bars = c.bars.map(function (b) { return { o: b[1], h: b[2], l: b[3], c: b[4] }; });
  return replayBracket(bars, times, c.start, c.opt);
});
out.align = input.align.map(function (a) {
  const parsedBars = parseBars(a.bars), trades = readExport(a.trades).table;
  const filter = a.filter ? { col: trades.header.indexOf(a.filter[0]), value: a.filter[1] } : null;
  const entries = tradeEntries(trades, filter);
  const al = alignEntries(entries, parsedBars.bars, parsedBars.barMs, 0.25);
  return { useUtc: al.useUtc, shift: al.shiftHours, matched: al.matched.length, entries: entries.length };
});
out.roots = input.roots.map(instrumentRoot);
out.wide = input.wide.map(function (text) {
  const p = parseBars(text);
  return p ? p.bars.map(function (b) { return [b.clock, b.utc, b.o, b.h, b.l, b.c]; }) : null;
});
console.log(JSON.stringify(out));
