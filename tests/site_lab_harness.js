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
// Roll-adjusted trades: the example ORB trades as a TradingView continuous contract would list
// them, one 15-minute chart bar early and a constant gap above real prices on each side of a roll.
(function () {
  const orig = tradeEntries(example, { col: example.header.indexOf("Strategy"), value: "ORB" });
  const roll = Date.UTC(2026, 8, 18), early = 15 * 60000;
  const moved = orig.map(function (e) {
    const gap = e.entry.clock < roll ? input.gaps[0] : input.gaps[1];
    return Object.assign({}, e, {
      entry: { clock: e.entry.clock - early, utc: NaN }, exit: { clock: e.exit.clock - early, utc: NaN },
      entryPrice: e.entryPrice + gap, exitPrice: e.exitPrice + gap
    });
  });
  const spec = INSTRUMENTS.ES, a0 = alignEntries(orig, parsed.bars, parsed.barMs, spec.tick), a1 = alignEntries(moved, parsed.bars, parsed.barMs, spec.tick);
  const opt = { spec: spec, stops: [6, 12, 20], targetTicks: 24, breakevenTicks: 12, maxHoldMs: 240 * 60000, gapMs: 30 * 60000, setups: 5, feePerContract: 4.12 };
  const values = function (a) { return stopScan(a, parsed.bars, opt).rows.map(function (r) { return r.values; }); };
  const noise = input.noise.map(function (n) {
    return { entry: { clock: n[0], utc: NaN }, exit: { clock: n[0] + 600000, utc: NaN }, side: n[2], entryPrice: n[1], exitPrice: n[1], qty: 1, symbol: "" };
  });
  const a2 = alignEntries(noise, parsed.bars, parsed.barMs, spec.tick);
  out.rolled = {
    plain: { adjusted: !!a0.adjusted, matched: a0.matched.length },
    adjusted: !!a1.adjusted, shiftMinutes: a1.shiftMinutes, matched: a1.matched.length, entries: moved.length,
    atOpen: a1.matched.filter(function (m) { return m.atOpen; }).length, offsets: a1.offsets.map(function (o) { return [o.gap, o.n]; }),
    before: values(a0), after: values(a1),
    noise: { adjusted: !!a2.adjusted, matched: a2.matched.length }
  };
})();
out.roots = input.roots.map(instrumentRoot);
out.names = input.names.map(instrumentFromName);
out.wide = input.wide.map(function (text) {
  const p = parseBars(text);
  return p ? p.bars.map(function (b) { return [b.clock, b.utc, b.o, b.h, b.l, b.c]; }) : null;
});
console.log(JSON.stringify(out));
