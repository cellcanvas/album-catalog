import React from "react";
import { Helmet } from "react-helmet";
import Header from "gatsby-theme-album/src/components/header";
import "@fontsource/ubuntu";
import "gatsby-theme-album/src/css/layout.css";
import "gatsby-theme-album/src/css/base-theme.css";
import "gatsby-theme-album/src/css/theme.css";

const Layout = ({ site, children }) => {
    const reversedChildren = React.Children.toArray(children).reverse();
    
  return (
    <>
      <Helmet
        title={site.siteMetadata.title}
        meta={[
          { name: 'description', content: 'sample description' },
          { name: 'keywords', content: 'sample, album, collection' },
        ]}
      >
      </Helmet>
      <div className="root" style={{ backgroundColor: "#c5d6d6" }}>
        <Header siteMeta={site.siteMetadata} />
        <div className="main">
          <div className="content">
            {reversedChildren}
          </div>
        </div>
      </div>
    </>
  )
}

export default Layout;
