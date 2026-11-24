// Test helper for tests/test_site_compare.py: runs the page's comparison on given pairs.
// The test prepends the page's own code (statistics, importer and comparison) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
console.log(JSON.stringify(input.pairs.map(function (p) { return compareGroups(p[0], p[1]); })));
