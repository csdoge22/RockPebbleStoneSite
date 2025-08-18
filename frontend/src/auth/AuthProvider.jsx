import { createContext, useContext, useEffect, useMemo, useState } from "react";
import axios from "axios";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const isJWTValid = (token) => {
        if (!token) return false;
        try {
            const [header, payload, signature] = token.split('.');
            if (!header || !payload || !signature) return false;
            
            const decodedPayload = JSON.parse(atob(payload));
            return decodedPayload.exp * 1000 > Date.now();
        } catch {
            return false;
        }
    };

    const [token, setToken] = useState(() => {
        const storedToken = localStorage.getItem("token");
        return isJWTValid(storedToken) ? storedToken : null;
    });

    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    const apiCall = async (config) => {
        try {
            setError(null);
            return await axios(config);
        } catch (err) {
            setError(err.response?.data?.message || "An error occurred");
            throw err;
        }
    };

    const fetchUserData = async () => {
        try {
            const response = await apiCall({
                method: 'get',
                url: '/api/user'
            });
            setUser(response.data);
        } catch (error) {
            if (error.response?.status === 401) {
                handleLogout();
            }
        }
    };

    const modifyToken = async (newToken) => {
        if (!isJWTValid(newToken)) {
            handleLogout();
            return;
        }
        setToken(newToken);
        await fetchUserData();
    };

    const handleLogout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem("token");
        delete axios.defaults.headers.common['Authorization'];
    };

    useEffect(() => {
        const initializeAuth = async () => {
            if (token) {
                axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
                localStorage.setItem("token", token);
                await fetchUserData();
            }
            setIsLoading(false);
        };

        initializeAuth();
    }, [token]); // Added token as dependency

    const contextValue = useMemo(() => ({
        token,
        user,
        isLoading,
        error,
        modifyToken,
        logout: handleLogout,
        setError
    }), [token, user, isLoading, error]);

    return (
        <AuthContext.Provider value={contextValue}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
};

export default AuthProvider;