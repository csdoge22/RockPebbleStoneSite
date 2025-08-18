import axios from "axios";
export const login = async (credentials) => {
  try {
    const response = await axios.post('http://localhost:8080/backend/api/auth/login', credentials);
    return {
      success: true,
      token: response.data.token, // Ensure your backend returns { token: "xyz" }
      user: response.data.user    // Optional: if you want user data immediately
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.message || 'Login failed'
    };
  }
};