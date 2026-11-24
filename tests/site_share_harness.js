// Test helper for tests/test_site_share.py: runs the page's share-link functions.
// The test prepends the page's own code (statistics, importer, share) to this file.
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));

function verdict(stats, n, setups) {
  const need = tradesNeeded(stats, setups, 0.95);
  return {
    prob: edgeProbability(stats, n, setups),
    need: isFinite(need) ? Math.ceil(need) : null,
    luck: luckPerTrade(stats, n, setups)
  };
}

const out = {
  // A list of trades sent through a link and read back.
  lists: input.lists.map(function (c) {
    const sent = shareFromTrades(c.values, c.before, c.unit, c.setups), text = encodeShare(sent), got = decodeShare("#" + text);
    return {
      text: text,
      sent: Object.assign(verdict(sent.stats, sent.trades, c.setups), {
        wins: sent.wins, losses: sent.losses, breakevens: sent.breakevens, maxDrawdown: sent.maxDrawdown,
        net: c.values.reduce(function (a, b) { return a + b; }, 0)
      }),
      got: Object.assign(verdict(got.stats, got.trades, got.setups), {
        kind: got.kind, trades: got.trades, setups: got.setups, unit: got.unit, wins: got.wins, losses: got.losses,
        breakevens: got.breakevens, maxDrawdown: got.maxDrawdown, net: got.stats.mean * got.trades
      })
    };
  }),
  quick: input.quick.map(function (q) { const text = encodeShare(q); return { text: text, got: decodeShare(text) }; }),
  decode: input.decode.map(decodeShare)
};
console.log(JSON.stringify(out));
