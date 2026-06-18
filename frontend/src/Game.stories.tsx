import type { Meta, StoryObj } from "@storybook/react";
import { Game } from "./Game";

const meta: Meta<typeof Game> = {
  title: "Game",
  component: Game,
};
export default meta;

type Story = StoryObj<typeof Game>;

export const Default: Story = {};