// Prints {dc, token}: a fresh OAuth access token of the signed-in Catalyst CLI (`catalyst login`).
// Used by upload_dataset.py; the token is only ever passed through a pipe, never stored.
const lib = require("path").join(process.env.APPDATA || require("os").homedir() + "/.npm-global",
  "npm/node_modules/zcatalyst-cli/lib");
const store = require(lib + "/util_modules/config-store.js").default;
const { getActiveDC } = require(lib + "/util_modules/dc.js");
const Credential = require(lib + "/authentication/credential.js").default;

(async () => {
  const dc = getActiveDC();
  const saved = store.get(dc + ".credential", null);
  if (!saved) throw new Error("the Catalyst CLI is not logged in for DC " + dc + " (run: catalyst login)");
  Credential.init(saved);
  const token = await Credential.getAccessToken();
  process.stdout.write(JSON.stringify({ dc, token }));
})().catch((e) => { console.error(String(e && e.message || e)); process.exit(1); });
