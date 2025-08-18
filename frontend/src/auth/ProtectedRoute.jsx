import { Navigate } from "react-router-dom";
import { useAuth } from "./AuthProvider";

export const ProtectedRoute = ({ children, redirectTo = '/login' }) => {
    const { token, isLoading } = useAuth();
    
    if (isLoading) return <div>Loading...</div>;
    if (!token) return <Navigate to={redirectTo} replace />;
    
    return children;
};