const API_URL = "http://localhost:8001";

export const predictSign = async (formData) => {
    const res = await fetch(`${API_URL}/predict/sign`, { method: "POST", body: formData });
    return await res.json();
};

export const predictVoice = async (formData) => {
    const res = await fetch(`${API_URL}/predict/voice`, { method: "POST", body: formData });
    return await res.json();
};
