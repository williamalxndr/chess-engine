import type { Meta, StoryObj } from "@storybook/react";
import { Cell } from "./Cell";

const meta: Meta<typeof Cell> = {
  title: "Cell",
  component: Cell,
};
export default meta;

type Story = StoryObj<typeof Cell>;

export const Empty: Story = { args: { value: 0, disabled: false } };
export const X: Story = { args: { value: -1, disabled: false } };
export const O: Story = { args: { value: 1, disabled: false } };
export const EmptyDisabled: Story = { args: { value: 0, disabled: true } };
export const XDisabled: Story = { args: { value: -1, disabled: true } };
export const ODisabled: Story = { args: { value: 1, disabled: true } };