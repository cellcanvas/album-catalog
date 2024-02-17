module.exports = {
  pathPrefix: `/`,
  siteMetadata: {
    title: 'cellcanvas album catalog',
    subtitle: 'sharing cellcanvas tools',
    catalog_url: 'https://github.com/cellcanvas/album-catalog',
    menuLinks:[
      {
         name:'Catalog',
         link:'/catalog'
      },
      {
         name:'About',
         link:'/about'
      },
    ]
  },
  plugins: [{ resolve: `gatsby-theme-album`, options: {} }],
}
