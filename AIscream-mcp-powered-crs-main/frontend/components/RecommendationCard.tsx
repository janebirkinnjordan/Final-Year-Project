type Props = {
  item: {
    domain: string;
    title: string;
    subtitle?: string;
    summary?: string;
    source?: string;
  };
};

export default function RecommendationCard({ item }: Props) {
  return (
    <div className="surface cardItem">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <strong>{item.title}</strong>
        <span className="badge">{item.domain}</span>
      </div>
      <div className="subtle" style={{ marginTop: 8 }}>{item.subtitle || '—'}</div>
      <p>{item.summary || 'No summary available.'}</p>
      <div className="subtle">Source: {item.source || 'Unknown'}</div>
    </div>
  );
}
