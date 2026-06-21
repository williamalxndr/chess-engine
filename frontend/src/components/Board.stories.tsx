import type { Meta, StoryObj } from "@storybook/react";
import { Board } from "./Board";
import type { CellValue } from "./Cell";

const meta: Meta<typeof Board> = {
  title: "Board",
  component: Board,
};
export default meta;

type Story = StoryObj<typeof Board>;

const empty: CellValue[][] = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];

const midGame: CellValue[][] = [
  [-1, 0, 1],
  [0, -1, 0],
  [1, 0, 0],
];

const fullBoard: CellValue[][] = [
  [-1, 1, -1],
  [1, -1, 1],
  [-1, 1, -1],
];

export const Empty: Story = {
  args: { board: empty, disabled: false },
};

export const MidGame: Story = {
  args: { board: midGame, disabled: false },
};

export const Disabled: Story = {
  args: { board: midGame, disabled: true },
};

export const Full: Story = {
  args: { board: fullBoard, disabled: true },
};