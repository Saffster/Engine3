// make_array.js
// Usage: node make_array.js tmdb_movies_with_meshes.json movies.array.js

import fs from "node:fs";

const [,, inPath = "tmdb_movies_with_meshes.json", outPath = "movies.array.js"] = process.argv;

const raw = fs.readFileSync(inPath, "utf-8");
const data = JSON.parse(raw);

if (!Array.isArray(data)) {
  throw new Error(`Expected top-level JSON array, got: ${typeof data}`);
}

const js = `// AUTO-GENERATED from ${inPath}\nexport default ${JSON.stringify(data, null, 2)};\n`;
fs.writeFileSync(outPath, js, "utf-8");

console.log(`Wrote ${outPath} (${data.length} items)`);
