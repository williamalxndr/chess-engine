import type { MoveRequest, MoveResponse } from "@/types";

const PREFIX = "/api"

async function request<T>(endpoint: string, options?: RequestInit): Promise<T>{
    const response = await fetch(`${PREFIX}${endpoint}`, {
        "headers": {
            "Content-Type": "application/json",
            ...options?.headers,
        },
        ...options
    });

    if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
    }

    const json = await response.json();   
    return json.data;  
}

export const api = {
    getMove: (data: MoveRequest, game: string) => 
        request<MoveResponse>(`/games/${game}/move`,
            {
                method: "POST",
                body: JSON.stringify(data)
            }
        ),
}