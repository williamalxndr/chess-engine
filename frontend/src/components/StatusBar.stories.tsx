import type { Meta, StoryObj } from "@storybook/react";
import { StatusBar } from "./StatusBar";

const meta: Meta<typeof StatusBar> = {
  title: "StatusBar",
  component: StatusBar,
};
export default meta;

type Story = StoryObj<typeof StatusBar>;

const score = { human: 2, bot: 1 };

export const Idle: Story = { args: { status: "idle", score, onSelectFirst: () => {} } };
export const HumanTurn: Story = { args: { status: "human_turn", score, onSelectFirst: () => {} } };
export const BotThinking: Story = { args: { status: "bot_thinking", score, onSelectFirst: () => {} } };
export const Win: Story = { args: { status: "win", score, onSelectFirst: () => {} } };
export const Lose: Story = { args: { status: "lose", score, onSelectFirst: () => {} } };
export const Draw: Story = { args: { status: "draw", score, onSelectFirst: () => {} } };