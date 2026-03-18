import Link from "next/link";

export default function HomePage() {
  return (
    <main style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h1>Hotel Review AI</h1>
      <p>Hotel review exploration + RAG QA.</p>
      <ul>
        <li>
          <Link href="/dashboard">Dashboard</Link>
        </li>
        <li>
          <Link href="/qa">Q&amp;A</Link>
        </li>
      </ul>
    </main>
  );
}

