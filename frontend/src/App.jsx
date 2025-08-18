import './App.css';
import AuthProvider from './auth/AuthProvider';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import { ProtectedRoute } from './auth/ProtectedRoute';
import Layout from './pages/Layout';
import Home from './pages/Home';
import Login from './pages/Login';
import Board from './pages/Board';
import Register from './pages/Register';
import About from './pages/About';
import Logout from './pages/Logout';

const router = createBrowserRouter([
  {
    element: <Layout />, // Navbar is included here exactly once
    children: [
      {
        path: "/",
        element: <Home />,
      },
      {
        path: "/login",
        element: <Login />,
      },
      {
        path: "/register",
        element: <Register />,
      },
      {
        path: "/about",
        element: <About />,
      },
      {
        path: "/board",
        element: (
          <ProtectedRoute>
            <Board />
          </ProtectedRoute>
        ),
      },
      {
        path: "/logout",
        element: (
          <ProtectedRoute>
            <Logout />
          </ProtectedRoute>
        ),
      },
    ],
  },
]);

function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
}

export default App;