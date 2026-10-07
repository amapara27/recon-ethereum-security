// Vector trace of public/logo.png: letters follow currentColor, the radar "o" takes `accent`.
const LETTERS = 'M4 94 l0 -86 19 0 18 0 0 29 0 28 15 0 16 0 22 -29 21 -28 22 0 c18 0 22 0 21 1 -1 3 -21 27 -30 40 -5 6 -11 14 -13 17 l-4 5 7 2 c22 8 33 25 33 50 0 20 -7 34 -22 44 -16 11 -22 12 -79 12 l-46 0 0 -85z m94 51 c17 -6 22 -27 10 -40 -6 -6 -11 -7 -40 -8 l-27 0 0 24 0 25 26 0 c20 0 27 0 31 -1z M212 137 c-26 -6 -45 -25 -52 -50 -2 -10 -1 -29 2 -39 10 -27 35 -44 67 -43 22 0 41 8 53 23 l3 4 -13 9 -13 8 -5 -4 c-14 -13 -38 -13 -51 0 -4 3 -9 13 -8 14 0 1 21 1 47 1 36 0 48 0 48 1 2 3 0 24 -2 32 -6 20 -21 35 -40 42 -9 3 -27 4 -36 2z m27 -32 c3 -1 7 -3 9 -5 4 -4 9 -14 8 -15 0 -1 -14 -1 -31 -1 l-31 0 1 2 c5 17 27 27 44 19z M355 137 c-53 -12 -73 -75 -36 -112 31 -31 86 -26 106 9 l3 6 -14 6 c-7 4 -14 7 -14 7 -1 0 -3 -2 -6 -5 -12 -14 -33 -15 -47 -2 -14 12 -14 38 0 50 8 7 12 9 23 9 13 0 19 -4 27 -13 l2 -3 13 7 c9 4 14 7 14 8 0 3 -7 13 -13 19 -14 12 -38 18 -58 14z M650 136 c-10 -2 -20 -8 -25 -15 l-3 -4 0 9 0 9 -18 0 -19 0 0 -63 0 -64 19 0 18 0 0 38 c0 37 0 39 2 43 10 21 40 22 50 1 1 -3 2 -8 2 -43 l0 -39 19 0 18 0 0 45 c-1 49 -1 48 -7 60 -10 19 -33 29 -56 23z'

export default function Wordmark({ className = '', accent = '#55cca7' }) {
  return (
    <svg className={`block ${className}`} height="20" viewBox="4 4 708 174" role="img" aria-label="Recon">
      <path transform="matrix(1 0 0 -1 0 182)" fill="currentColor" d={LETTERS} />
      <g fill="none" stroke={accent}>
        <circle cx="500" cy="110" r="66" strokeWidth="7" />
        <circle cx="500" cy="110" r="42" strokeWidth="6" opacity=".5" />
        <path d="M517 93 562 48" strokeWidth="11" />
      </g>
      <circle cx="500" cy="110" r="14.5" fill={accent} />
    </svg>
  )
}
