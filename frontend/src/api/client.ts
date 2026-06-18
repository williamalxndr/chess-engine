import type { MoveRequest, MoveResponse } from "@/types";

const BASE_URL = import.meta.env.VITE_API_URL;

async function request<T>(endpoint: string, options?: RequestInit): Promise<T>{
    const response = await fetch(`${BASE_URL}${endpoint}`, {
        "headers": {
            "Content-Type": "application/json",
            ...options?.headers,
        },
        ...options
    });

    if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
    }
    return response.json()
}

export const api = {
    move: (data: MoveRequest, game: string) => 
        request<MoveResponse>(`games/${game}/move`,
            {
                method: "POST",
                body: JSON.stringify(data)
            }
        ),
}