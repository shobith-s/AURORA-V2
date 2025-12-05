/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // New minimal palette
        'brand': {
          black: '#000000',
          white: '#FFFFFF',
          'warm-gray': '#E0E0E0',
          'cool-gray': '#9E9E9E',
          'soft-blue': '#90CAF9',
        },
        // Semantic colors using the palette
        primary: {
          DEFAULT: '#90CAF9',
          hover: '#64B5F6',
          light: '#BBDEFB',
          dark: '#42A5F5',
        },
        background: {
          DEFAULT: '#FFFFFF',
          dark: '#000000',
          muted: '#E0E0E0',
        },
        foreground: {
          DEFAULT: '#000000',
          muted: '#9E9E9E',
          inverse: '#FFFFFF',
        },
        border: {
          DEFAULT: '#E0E0E0',
          dark: '#9E9E9E',
        },
        // Status colors (derived from palette)
        success: '#4CAF50',
        warning: '#FFC107',
        error: '#F44336',
        info: '#90CAF9',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'gradient': 'gradient 8s ease infinite',
      },
      keyframes: {
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          },
        },
      },
    },
  },
  plugins: [],
}
