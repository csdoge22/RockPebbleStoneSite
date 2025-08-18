import { Link } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";

const Navbar = () => {
    const { token, user, logout, isLoading } = useAuth();

    if (isLoading) return <div className="navbar-loading">Loading...</div>;

    return (
        <nav className="navbar bg-green-700 p-4 text-white">
            <div className="container mx-auto flex justify-between items-center">
                <div className="flex space-x-4">
                    <Link to="/" className="hover:text-green-300">Home</Link>
                    <Link to="/about" className="hover:text-green-300">About</Link>
                    {token && <Link to="/board" className="hover:text-green-300">Board</Link>}
                </div>
                
                <div className="flex space-x-4 items-center">
                    {token ? (
                        <>
                            <span className="text-sm">Welcome, {user?.name}</span>
                            <button 
                                onClick={logout}
                                className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded"
                            >
                                Logout
                            </button>
                        </>
                    ) : (
                        <>
                            <Link to="/login" className="hover:text-green-300">Login</Link>
                            <Link to="/register" className="hover:text-green-300">Register</Link>
                        </>
                    )}
                </div>
            </div>
        </nav>
    );
};

export default Navbar;