/** Deterministic hash for identicon generation. */
export function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

/** GitHub-style mirrored 5×5 block identicon from a stable seed (user id, club id, etc.). */
export function Identicon({
  seed,
  className,
  title,
}: {
  seed: string;
  className?: string;
  title?: string;
}) {
  const hash = hashString(seed);
  const hue = hash % 360;
  const bg = `hsl(${hue}, 44%, 50%)`;

  const pattern: boolean[] = [];
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 3; col++) {
      const bit = (hash >> (row * 3 + col)) & 1;
      const alt = (hash >> (row * 3 + col + 9)) & 1;
      pattern.push(bit === 1 || alt === 1);
    }
  }

  const blocks = [];
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      const sourceCol = col < 3 ? col : 4 - col;
      if (!pattern[row * 3 + sourceCol]) continue;
      blocks.push(
        <rect key={`${row}-${col}`} x={col} y={row} width={1} height={1} fill="#f0f3f6" fillOpacity={0.92} />,
      );
    }
  }

  return (
    <svg
      viewBox="0 0 5 5"
      shapeRendering="crispEdges"
      className={className}
      role="img"
      aria-label={title ?? "Default avatar"}
    >
      <title>{title ?? "Default avatar"}</title>
      <rect width={5} height={5} fill={bg} />
      {blocks}
    </svg>
  );
}
