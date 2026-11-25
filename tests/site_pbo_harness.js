// Test helper for tests/test_site_pbo.py: runs the page's port of the PBO check.
// The test prepends the page's own code (the pbo block) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
console.log(JSON.stringify({
  cases: input.cases.map(function (c) { return pboCSCV(c.matrix, c.s); }),
  ranks: input.ranks.map(averageRanks),
  splits: input.splits.map(function (s) { return halfSplits(s).length; }),
  periods: input.periods.map(pboPeriods),
  daily: input.daily.map(function (d) { return dailyMatrix(d.versions, d.days, d.s); }),
  bad: input.bad.map(function (c) { return pboCSCV(c.matrix, c.s); })
}));
