import js from "@eslint/js";
import globals from "globals";

export default [
  {
    ignores: ["dist/**", "build/**"],
  },
  {
    files: ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: globals.browser,
    },
    rules: {
      semi: "warn",
    },
  },
];
