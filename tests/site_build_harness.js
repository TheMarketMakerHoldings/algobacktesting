// Test helper for tests/test_site_build.py: runs the page's strategy builder.
// The test prepends the page's own code (statistics, importer, lab and builder) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));

function run(bars, rule) {
  return buildTrades(bars, rule).map(function (t) {
    return [t.entryTime, t.side, t.entryPrice, t.exitPrice, t.exit, t.before, t.pnl, t.exitTime];
  });
}
const example = parseBars(input.exampleBars).bars;
console.log(JSON.stringify({
  random: input.rules.map(function (rule) { return run(input.bars, rule); }),
  cases: input.cases.map(function (c) { return run(c.bars, c.rule); }),
  example: input.exampleRules.map(function (rule) { return run(example, rule); }),
  problems: input.problems.map(ruleProblem),
  clocks: input.clocks.map(parseClock),
  sentences: input.sentences.map(describeRule),
  keys: input.keyRules.map(ruleKey),
  parsed: input.phrases.map(function (p) {
    const out = parseStrategy(p[0], p[1]);
    out.problem = ruleProblem(Object.assign({ tick: 0.25, point: 50, fee: 0 }, out.rule));
    return out;
  })
}));
