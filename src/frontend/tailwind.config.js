/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                eagle: {
                    green: {
                        DEFAULT: '#00E054', // Neon Green
                        hover: '#00C244',
                        light: 'rgba(0, 224, 84, 0.1)',
                        glow: 'rgba(0, 224, 84, 0.5)',
                    },
                    black: {
                        DEFAULT: '#050505', // Deep Black
                        surface1: '#0F0F11',
                        surface2: '#18181B',
                        border: '#27272A',
                    },
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                display: ['Outfit', 'Inter', 'system-ui', 'sans-serif'],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'float': 'float 3s ease-in-out infinite',
            },
            keyframes: {
                float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                }
            }
        },
    },
    plugins: [],
}
