/**
 * The public interface of this module exposes the functions as specified by
 * https://github.com/ethereum/consensus-specs/blob/dev/specs/eip4844/polynomial-commitments.md#kzg
 */
const fs = require("fs");
const bindings = require("bindings")("kzg");

const originalLoadTrustedSetup = bindings.loadTrustedSetup;

// docstring in ./kzg.d.ts with exported definition
bindings.loadTrustedSetup = function loadTrustedSetup(filePath) {
  if (!(filePath && typeof filePath === "string")) {
    throw new TypeError("must initialize kzg with the filePath to a TSIF file");
  }
  if (!fs.existsSync(filePath)) {
    throw new Error(`no trusted setup found: ${filePath}`);
  }
  originalLoadTrustedSetup(filePath);
};

module.exports = exports = bindings;
