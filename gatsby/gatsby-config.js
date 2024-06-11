module.exports = {
  pathPrefix: `/`,
  siteMetadata: {
    title: 'cellcanvas album catalog',
    subtitle: 'sharing cellcanvas tools',
    catalog_url: 'https://github.com/cellcanvas/album-catalog',
    menuLinks:[        
        {
            name:'About',
            link:'/about'
        },
        {
            name:'Solution Catalog',
            link:'/catalog'
        },
        {
            name:'Documentation',
            link:'/documentation'
        },
        {
            name:'Tutorial',
            link:'/tutorial'
        },
    ]
  },
  plugins: [{ resolve: `gatsby-theme-album`, options: {} }],
}
