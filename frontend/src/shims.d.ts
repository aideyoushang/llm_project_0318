declare module "next/link" {
  const Link: any;
  export default Link;
}

declare module "react" {
  export const useMemo: any;
  export const useState: any;
}

declare const process: {
  env: Record<string, string | undefined>;
};

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}
