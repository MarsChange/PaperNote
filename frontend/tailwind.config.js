/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#fffdf8',
          secondary: '#f5eee2',
          tertiary: '#ece1d0',
        },
        border: {
          DEFAULT: '#decfb8',
          light: '#eadfce',
        },
        text: {
          primary: '#241e18',
          secondary: '#5e5548',
          tertiary: '#8d816f',
        },
        accent: {
          DEFAULT: '#0f766e',
          hover: '#0a5d57',
          light: '#d8f1ee',
        },
      },
      fontFamily: {
        sans: ['"Avenir Next"', '"PingFang SC"', '"Noto Sans SC"', '"Segoe UI"', 'sans-serif'],
        display: ['"Iowan Old Style"', '"Palatino Linotype"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
    },
  },
  plugins: [],
}
