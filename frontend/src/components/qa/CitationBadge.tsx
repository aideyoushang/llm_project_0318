type Props = {
  index: number;
  reference: Record<string, unknown>;
};

export default function CitationBadge(props: Props) {
  const hotelId = props.reference["hotel_id"];
  const docId = props.reference["doc_id"];
  const labelParts = [props.index.toString()];
  if (hotelId) labelParts.push(String(hotelId));
  if (docId) labelParts.push(String(docId));
  const label = labelParts.join(":");

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "4px 8px",
        border: "1px solid #ddd",
        borderRadius: 999,
        fontSize: 12
      }}
      title={JSON.stringify(props.reference)}
    >
      {label}
    </span>
  );
}

