export interface MoveRequest {
    board: number[][]
}

export interface MoveResponse {
    move: number | null,
    game_over: boolean,
    result: number | null
}
