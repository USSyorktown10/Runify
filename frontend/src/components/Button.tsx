import "@/styles/button.css";

type ButtonProps = {
  onClick: () => void;
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
  size?: 'small' | 'medium' | 'large';
  type?: 'submit' | 'reset' | 'button';
};

export function Button({ onClick, children, variant = 'primary', size = 'medium', type = 'button' }: ButtonProps) {
  return (
    <button onClick={onClick} className={`btn btn-${variant} btn-${size}`} type={type}>
      {children}
    </button>
  );
}