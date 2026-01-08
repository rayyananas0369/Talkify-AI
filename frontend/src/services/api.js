const API_URL = "http://localhost:8000";

export const predictSign = async (formData) => {
    const res = await fetch(`${API_URL}/predict/sign`, { method: "POST", body: formData });
    return await res.json();
};

export const predictLip = async (formData) => {
    const res = await fetch(`${API_URL}/predict/lip`, { method: "POST", body: formData });
    return await res.json();
};
