// Test helper for tests/test_site_saved.py: runs the page's saved-strategy functions.
// The test prepends the page's own code (statistics, importer, comparison, saved) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const fns = {
  cleanName, makeSaved, addSaved, removeSaved, serializeSaved, parseSaved, sameTrades,
  savedByName, savedById, statsFromTrades, compareGroups
};
const out = { calls: input.calls.map(function (c) { return fns[c.fn].apply(null, c.args); }) };

// Round trip: save each list under its name, write the whole set as text, read it back.
let list = [];
input.roundtrip.forEach(function (r) {
  list = addSaved(list, makeSaved(r.name, r.values, r.unit, r.setups, r.now).item).list;
});
const text = serializeSaved(list), back = parseSaved(text);
out.roundtrip = {
  list: list,
  back: back,
  bytes: text.length,
  stats: back.map(function (s) { return statsFromTrades(s.values); }),
  compare: compareGroups(back[0].values, back[1].values)
};
console.log(JSON.stringify(out));
