import * as React from "react"
import { StaticImage } from "gatsby-plugin-image"

export default function Logo() {
  return (
  <StaticImage className="logo" src="../images/logo.png" alt="catalog logo" placeholder="blurred" fit="contain"/>
  );
}