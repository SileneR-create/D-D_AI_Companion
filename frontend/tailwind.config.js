/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        dnd: {
          brown: '#7A5C3E',
          'brown-dark': '#3D2B1F',
          'brown-light': '#A67C5B',
          cream: '#F7F3E9',
          tan: '#EDE5D8',
          dark: '#2B2B2B',
          red: '#8B0000',
        },
      },
      fontFamily: {
        display: ['Cinzel', 'serif'],
        body: ['Crimson Text', 'Georgia', 'serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
