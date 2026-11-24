// Test helper for tests/test_site_when.py: runs the page's timing breakdown on given cases.
// The test prepends the page's own code (statistics, importer and timing) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
console.log(JSON.stringify({
  groups: input.groups.map(function (c) { return groupByTime(c.values, c.times, c.by); }),
  slices: input.slices.map(function (c) { return timeSlices(c.values, c.times, c.by, c.setups, c.minTrades); })
}));
