import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        peach: {
          DEFAULT: "#FFB347",
          light: "#FFCC99",
          dark: "#E79523",
        },
      },
    },
  },
  plugins: [],
};

export default config;
