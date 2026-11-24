// Test helper for tests/test_site_prop.py: runs the page's prop-firm simulator on given cases.
// The test prepends the page's own code (statistics, importer and simulator) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
console.log(JSON.stringify({
  sims: input.sims.map(function (c) { return simulateEvaluation(c.values, c.rules, c.opts); }),
  perDay: input.perDay.map(tradesPerDay)
}));
